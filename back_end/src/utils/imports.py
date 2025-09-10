"""
Centralized import utilities and optional dependency management.
"""
import importlib
import logging
import sys
from typing import Optional, Any, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class ImportManager:
    """Manages optional imports and fallback mechanisms."""
    
    def __init__(self):
        self._import_cache: Dict[str, Any] = {}
        self._failed_imports: List[str] = []
    
    def try_import(self, module_name: str, from_module: Optional[str] = None, 
                   fallback: Optional[Any] = None) -> Optional[Any]:
        """
        Try to import a module with fallback handling.
        
        Args:
            module_name: Name of module/class to import
            from_module: Parent module name
            fallback: Fallback value if import fails
            
        Returns:
            Imported module/class or fallback
        """
        cache_key = f"{from_module}.{module_name}" if from_module else module_name
        
        if cache_key in self._import_cache:
            return self._import_cache[cache_key]
        
        if cache_key in self._failed_imports:
            return fallback
        
        try:
            if from_module:
                module = importlib.import_module(from_module)
                imported = getattr(module, module_name)
            else:
                imported = importlib.import_module(module_name)
            
            self._import_cache[cache_key] = imported
            logger.debug(f"Successfully imported {cache_key}")
            return imported
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to import {cache_key}: {e}")
            self._failed_imports.append(cache_key)
            self._import_cache[cache_key] = fallback
            return fallback
    
    def get_enhanced_components(self) -> Dict[str, Any]:
        """Get all enhanced components with fallbacks."""
        components = {}
        
        # Try enhanced imports first
        components['database_manager'] = self.try_import(
            'database_manager', 'back_end.src.data.database_manager_enhanced'
        )
        components['pubmed_collector'] = self.try_import(
            'EnhancedPubMedCollector', 'back_end.src.data.pubmed_collector_enhanced'
        )
        components['probiotic_analyzer'] = self.try_import(
            'EnhancedProbioticAnalyzer', 'back_end.src.data.probiotic_analyzer_enhanced'
        )
        
        return {k: v for k, v in components.items() if v is not None}
    
    def check_requirements(self, requirements: List[str]) -> Dict[str, bool]:
        """Check if required modules are available."""
        status = {}
        for req in requirements:
            try:
                importlib.import_module(req)
                status[req] = True
            except ImportError:
                status[req] = False
        return status


def import_optional_dependency(name, extra=""):
    """
    Import an optional dependency.
    
    Args:
        name: The module name
        extra: Extra installation info if import fails
    
    Returns:
        The imported module or None if not available
    """
    try:
        return __import__(name)
    except ImportError:
        print(f"Optional dependency '{name}' not available. {extra}")
        return None


# Global import manager
import_manager = ImportManager()