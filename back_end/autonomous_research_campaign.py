#!/usr/bin/env python3
"""
Autonomous Research Campaign - Long-running multi-condition research orchestrator
Handles paper collection and LLM processing with full fault tolerance and thermal protection.
"""

import sys
import json
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add the src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_dir))

from src.data.config import config, setup_logging
from src.paper_collection.database_manager import database_manager
from src.paper_collection.pubmed_collector import PubMedCollector
from src.llm.dual_model_analyzer import DualModelAnalyzer

logger = setup_logging(__name__, 'autonomous_campaign.log')


@dataclass
class CampaignConfig:
    """Configuration for autonomous research campaign."""
    conditions: List[str]
    papers_per_condition: int = 100
    min_year: int = 2015
    max_temp_celsius: int = 75
    cooling_temp_celsius: int = 65
    thermal_check_interval: int = 30  # seconds
    retry_delays: List[int] = None  # [30, 60, 120, 300] seconds
    max_retries_per_condition: int = 3
    enable_s2_enrichment: bool = True
    batch_size: int = 5

    def __post_init__(self):
        if self.retry_delays is None:
            self.retry_delays = [30, 60, 120, 300]


@dataclass
class ConditionProgress:
    """Progress tracking for a single condition."""
    condition: str
    status: str  # 'pending', 'collecting', 'processing', 'completed', 'failed'
    papers_collected: int = 0
    papers_processed: int = 0
    target_papers: int = 100
    collection_start_time: Optional[str] = None
    processing_start_time: Optional[str] = None
    completion_time: Optional[str] = None
    retry_count: int = 0
    last_error: Optional[str] = None


class ThermalProtectionManager:
    """Manages GPU thermal protection with automatic cooling."""

    def __init__(self, max_temp: int = 75, cooling_temp: int = 65):
        self.max_temp = max_temp
        self.cooling_temp = cooling_temp
        self.is_cooling = False

    def check_temperature(self) -> tuple[bool, float]:
        """Check GPU temperature and return (is_safe, current_temp)."""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu',
                                  '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                temp = float(result.stdout.strip())
                is_safe = temp <= self.max_temp
                return is_safe, temp
            else:
                logger.warning("Could not read GPU temperature, assuming safe")
                return True, 0.0
        except Exception as e:
            logger.warning(f"GPU temperature check failed: {e}")
            return True, 0.0

    def wait_for_cooling(self) -> None:
        """Wait for GPU to cool down to safe temperature."""
        logger.info(f"🌡️  GPU overheated! Waiting for cooling to {self.cooling_temp}°C...")
        self.is_cooling = True

        while True:
            is_safe, current_temp = self.check_temperature()

            if current_temp <= self.cooling_temp:
                logger.info(f"🌡️  GPU cooled to {current_temp}°C. Resuming operations.")
                self.is_cooling = False
                break

            logger.info(f"🌡️  Current GPU temperature: {current_temp}°C. Waiting...")
            time.sleep(30)


class CampaignSessionManager:
    """Manages campaign session persistence and recovery."""

    def __init__(self, session_file: str = "campaign_session.json"):
        self.session_file = Path(session_file)
        self.session_data = self._load_session()

    def _load_session(self) -> Dict[str, Any]:
        """Load existing session or create new one."""
        if self.session_file.exists():
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                logger.info(f"📋 Loaded existing campaign session: {len(data.get('conditions', {}))} conditions")
                return data
            except Exception as e:
                logger.error(f"Failed to load session file: {e}")

        # Create new session
        logger.info("📋 Creating new campaign session")
        return {
            'campaign_id': datetime.now().isoformat(),
            'start_time': datetime.now().isoformat(),
            'conditions': {},
            'global_status': 'active',
            'last_update': datetime.now().isoformat()
        }

    def save_session(self) -> None:
        """Save current session state."""
        self.session_data['last_update'] = datetime.now().isoformat()
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def get_condition_progress(self, condition: str) -> ConditionProgress:
        """Get progress for a specific condition."""
        if condition not in self.session_data['conditions']:
            self.session_data['conditions'][condition] = {
                'condition': condition,
                'status': 'pending',
                'papers_collected': 0,
                'papers_processed': 0,
                'target_papers': 100,
                'retry_count': 0
            }

        data = self.session_data['conditions'][condition]
        return ConditionProgress(**data)

    def update_condition_progress(self, progress: ConditionProgress) -> None:
        """Update progress for a condition."""
        self.session_data['conditions'][progress.condition] = {
            'condition': progress.condition,
            'status': progress.status,
            'papers_collected': progress.papers_collected,
            'papers_processed': progress.papers_processed,
            'target_papers': progress.target_papers,
            'collection_start_time': progress.collection_start_time,
            'processing_start_time': progress.processing_start_time,
            'completion_time': progress.completion_time,
            'retry_count': progress.retry_count,
            'last_error': progress.last_error
        }
        self.save_session()

    def get_campaign_summary(self) -> Dict[str, Any]:
        """Get overall campaign progress summary."""
        conditions = self.session_data.get('conditions', {})
        total_conditions = len(conditions)
        completed = len([c for c in conditions.values() if c['status'] == 'completed'])
        failed = len([c for c in conditions.values() if c['status'] == 'failed'])
        in_progress = len([c for c in conditions.values() if c['status'] in ['collecting', 'processing']])

        return {
            'total_conditions': total_conditions,
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'pending': total_conditions - completed - failed - in_progress,
            'completion_rate': (completed / total_conditions * 100) if total_conditions > 0 else 0
        }


class AutonomousResearchCampaign:
    """Main orchestrator for long-running multi-condition research campaigns."""

    def __init__(self, config: CampaignConfig):
        self.config = config
        self.session = CampaignSessionManager()
        self.thermal_manager = ThermalProtectionManager(
            max_temp=config.max_temp_celsius,
            cooling_temp=config.cooling_temp_celsius
        )

        # Initialize components
        self.collector = PubMedCollector()
        self.analyzer = DualModelAnalyzer()

        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"🚀 Autonomous Research Campaign initialized")
        logger.info(f"📋 Target conditions: {self.config.conditions}")
        logger.info(f"📊 Papers per condition: {self.config.papers_per_condition}")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown signals."""
        logger.info(f"🛑 Shutdown signal received ({signum}). Initiating graceful shutdown...")
        self.shutdown_requested = True

    def run_campaign(self) -> Dict[str, Any]:
        """Run the complete autonomous research campaign."""
        logger.info("🎯 Starting Autonomous Research Campaign")

        try:
            for condition in self.config.conditions:
                if self.shutdown_requested:
                    logger.info("🛑 Shutdown requested, stopping campaign")
                    break

                self._process_condition_with_retries(condition)

            # Campaign completion
            summary = self.session.get_campaign_summary()
            logger.info("🎉 Campaign completed!")
            logger.info(f"📊 Final summary: {summary}")

            return summary

        except Exception as e:
            logger.error(f"💥 Campaign failed with unexpected error: {e}")
            raise

    def _process_condition_with_retries(self, condition: str) -> None:
        """Process a single condition with retry logic."""
        progress = self.session.get_condition_progress(condition)

        # Skip if already completed
        if progress.status == 'completed':
            logger.info(f"✅ Condition '{condition}' already completed, skipping")
            return

        logger.info(f"🔬 Processing condition: {condition}")

        for attempt in range(self.config.max_retries_per_condition):
            try:
                if self.shutdown_requested:
                    return

                # Check thermal status before starting
                self._ensure_thermal_safety()

                # Collection phase
                if progress.status in ['pending', 'failed']:
                    self._collect_papers_for_condition(condition, progress)

                # Processing phase
                if progress.status == 'collecting':
                    self._process_papers_for_condition(condition, progress)

                # Mark as completed
                progress.status = 'completed'
                progress.completion_time = datetime.now().isoformat()
                self.session.update_condition_progress(progress)

                logger.info(f"✅ Condition '{condition}' completed successfully")
                return

            except Exception as e:
                attempt_num = attempt + 1
                progress.retry_count = attempt_num
                progress.last_error = str(e)

                logger.error(f"❌ Attempt {attempt_num} failed for '{condition}': {e}")

                if attempt_num < self.config.max_retries_per_condition:
                    delay = self.config.retry_delays[min(attempt, len(self.config.retry_delays) - 1)]
                    logger.info(f"⏳ Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    progress.status = 'failed'
                    logger.error(f"💀 Condition '{condition}' failed after {self.config.max_retries_per_condition} attempts")

                self.session.update_condition_progress(progress)

    def _collect_papers_for_condition(self, condition: str, progress: ConditionProgress) -> None:
        """Collect papers for a specific condition."""
        logger.info(f"📚 Collecting papers for: {condition}")

        progress.status = 'collecting'
        progress.collection_start_time = datetime.now().isoformat()
        self.session.update_condition_progress(progress)

        # Use existing collection functionality
        collection_result = self.collector.collect_interventions_by_condition(
            condition=condition,
            min_year=self.config.min_year,
            max_results=self.config.papers_per_condition,
            include_fulltext=True,
            use_interleaved_s2=self.config.enable_s2_enrichment
        )

        progress.papers_collected = collection_result.get('paper_count', 0)
        logger.info(f"📚 Collected {progress.papers_collected} papers for '{condition}'")

        # Move to processing status
        progress.status = 'processing'
        self.session.update_condition_progress(progress)

    def _process_papers_for_condition(self, condition: str, progress: ConditionProgress) -> None:
        """Process papers for a specific condition with thermal monitoring."""
        logger.info(f"🤖 Processing papers for: {condition}")

        progress.processing_start_time = datetime.now().isoformat()
        self.session.update_condition_progress(progress)

        # Get unprocessed papers for this condition
        unprocessed_papers = self.analyzer.get_unprocessed_papers()

        if not unprocessed_papers:
            logger.info(f"🤖 No papers to process for '{condition}'")
            return

        # Process in batches with thermal monitoring
        total_papers = len(unprocessed_papers)
        processed_count = 0

        for i in range(0, total_papers, self.config.batch_size):
            if self.shutdown_requested:
                return

            # Thermal check before each batch
            self._ensure_thermal_safety()

            batch = unprocessed_papers[i:i + self.config.batch_size]

            logger.info(f"🤖 Processing batch {i//self.config.batch_size + 1}: {len(batch)} papers")

            try:
                results = self.analyzer.process_papers_batch(
                    papers=batch,
                    save_to_db=True,
                    batch_size=len(batch)
                )

                processed_count += results.get('successful_papers', 0)
                progress.papers_processed = processed_count
                self.session.update_condition_progress(progress)

                logger.info(f"🤖 Processed {processed_count}/{total_papers} papers for '{condition}'")

            except Exception as e:
                logger.error(f"🤖 Batch processing failed: {e}")
                raise

    def _ensure_thermal_safety(self) -> None:
        """Ensure GPU temperature is safe before continuing."""
        is_safe, current_temp = self.thermal_manager.check_temperature()

        if not is_safe:
            logger.warning(f"🌡️  GPU temperature too high: {current_temp}°C")
            self.thermal_manager.wait_for_cooling()

    def get_status(self) -> Dict[str, Any]:
        """Get current campaign status."""
        summary = self.session.get_campaign_summary()

        # Add thermal status
        is_safe, current_temp = self.thermal_manager.check_temperature()
        summary['thermal_status'] = {
            'current_temp': current_temp,
            'is_safe': is_safe,
            'is_cooling': self.thermal_manager.is_cooling
        }

        return summary


def main():
    """Main entry point for autonomous research campaign."""
    parser = argparse.ArgumentParser(description='Autonomous Research Campaign')
    parser.add_argument('--conditions', nargs='+',
                       default=['ibs', 'ibd', 'gerd', 'constipation', 'diarrhea'],
                       help='Health conditions to research')
    parser.add_argument('--papers-per-condition', type=int, default=100,
                       help='Number of papers to collect per condition')
    parser.add_argument('--min-year', type=int, default=2015,
                       help='Minimum publication year')
    parser.add_argument('--max-temp', type=int, default=75,
                       help='Maximum GPU temperature before cooling')
    parser.add_argument('--cooling-temp', type=int, default=65,
                       help='Resume temperature after cooling')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='LLM processing batch size')
    parser.add_argument('--status-only', action='store_true',
                       help='Show status and exit')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing campaign')

    args = parser.parse_args()

    # Create campaign configuration
    campaign_config = CampaignConfig(
        conditions=args.conditions,
        papers_per_condition=args.papers_per_condition,
        min_year=args.min_year,
        max_temp_celsius=args.max_temp,
        cooling_temp_celsius=args.cooling_temp,
        batch_size=args.batch_size
    )

    # Initialize campaign
    campaign = AutonomousResearchCampaign(campaign_config)

    if args.status_only:
        status = campaign.get_status()
        print("📊 Campaign Status:")
        print(json.dumps(status, indent=2))
        return

    # Run the campaign
    try:
        results = campaign.run_campaign()
        print("🎉 Campaign completed successfully!")
        print(f"📊 Results: {results}")

    except KeyboardInterrupt:
        print("\n🛑 Campaign interrupted by user")
    except Exception as e:
        print(f"💥 Campaign failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()