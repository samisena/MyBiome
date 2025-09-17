# Robust LLM Paper Processor

An enhanced version of the LLM paper processing pipeline with advanced features for reliability, thermal protection, and progress recovery.

## Key Features

### 🔄 **Progress Saving & Recovery**
- **Automatic Checkpoints**: Saves progress after each paper (configurable, default: 1)
- **Session Recovery**: Resume processing from where you left off after interruptions
- **Graceful Shutdown**: Handles Ctrl+C and system signals safely
- **Progress Tracking**: JSON-based session files track detailed progress

### 🌡️ **Thermal Protection**
- **Real-time Monitoring**: Continuous GPU temperature and power monitoring
- **Automatic Pausing**: Stops processing when GPU gets too hot
- **Cooling Waits**: Waits for temperature to drop before resuming
- **Safety Thresholds**: Configurable temperature limits (default: 80°C max, 70°C resume)
- **Thermal Events Log**: Records all thermal events for analysis

### 💾 **Memory Optimization**
- **GPU Memory Monitoring**: Tracks VRAM usage during processing
- **Dynamic Batch Sizing**: Adjusts batch sizes based on available memory
- **Memory-aware Processing**: Prevents out-of-memory errors

### 🛡️ **Error Handling**
- **Robust Error Recovery**: Continues processing even if individual papers fail
- **Detailed Logging**: Comprehensive logs for debugging
- **Status Reporting**: Real-time status updates and summaries

## Usage Examples

### Basic Processing
```bash
# Process papers with default thermal protection
python robust_llm_processor.py --limit 50

# Custom batch size and temperature limits
python robust_llm_processor.py --limit 50 --batch-size 3 --max-temp 75
```

### Session Management
```bash
# Resume from previous interrupted session
python robust_llm_processor.py --resume

# Check current session status
python robust_llm_processor.py --session-status

# Clear session and start fresh
python robust_llm_processor.py --clear-session
```

### Monitoring
```bash
# Check current GPU thermal status
python robust_llm_processor.py --thermal-status

# Process with more frequent checkpoints
python robust_llm_processor.py --limit 20 --checkpoint-interval 3
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--limit N` | Maximum number of papers to process | All unprocessed |
| `--batch-size N` | Papers per batch | 5 |
| `--max-temp T` | Maximum GPU temperature (°C) | 80.0 |
| `--cooling-temp T` | Resume temperature after cooling (°C) | 70.0 |
| `--checkpoint-interval N` | Save progress every N papers | 1 |
| `--session-file FILE` | Custom session file path | processing_session.json |
| `--resume` | Resume previous session | False |
| `--session-status` | Show session status and exit | False |
| `--thermal-status` | Show thermal status and exit | False |
| `--clear-session` | Clear session and exit | False |

## Session Files

The processor creates a JSON session file that tracks:
- Session ID and timestamps
- Progress counters (total, processed, failed papers)
- Current batch information
- Thermal events log
- Configuration settings

Example session file structure:
```json
{
  "session_id": "20250917_192137",
  "start_time": 1694984497.123,
  "total_papers": 50,
  "processed_papers": 23,
  "failed_papers": ["12345", "67890"],
  "current_batch": 5,
  "batch_size": 5,
  "last_checkpoint": 1694984897.456,
  "interventions_extracted": 89,
  "session_config": {
    "max_temp": 80.0,
    "checkpoint_interval": 1
  },
  "thermal_events": [
    {
      "timestamp": 1694984600.0,
      "event_type": "thermal_warning",
      "temperature": 82.5,
      "power": 245.0,
      "max_temp": 80.0,
      "max_power": 250.0
    }
  ]
}
```

## Thermal Protection Details

### Temperature Monitoring
- **Continuous Monitoring**: Background thread checks GPU status every 5 seconds
- **Pre-processing Check**: Verifies safe temperature before starting
- **Batch-level Checks**: Monitors temperature between batches
- **Automatic Cooling**: Pauses processing and waits for temperature to drop

### Safety Thresholds
- **Maximum Temperature**: Default 80°C (configurable)
- **Cooling Resume**: Default 70°C (configurable)
- **Maximum Wait**: 5 minutes for cooling (prevents infinite waits)
- **Power Monitoring**: Optional power draw limits

### Thermal Events
The system logs thermal events including:
- Temperature warnings when limits are exceeded
- Cooling periods and their duration
- Power draw spikes
- Memory usage peaks

## Error Handling & Recovery

### Graceful Interruption
- **Signal Handling**: Captures Ctrl+C, SIGTERM gracefully
- **Progress Saving**: Saves current progress before shutdown
- **Clean Exit**: Proper cleanup of resources and monitoring threads

### Recovery Scenarios
1. **Power Loss**: Resume from last checkpoint on restart
2. **System Crash**: Session file preserves progress
3. **GPU Overheating**: Automatic cooling with resume capability
4. **Out of Memory**: Dynamic batch size adjustment
5. **Network Issues**: Individual paper failures don't stop the entire process

### Logging
- **Comprehensive Logs**: All events logged to `robust_llm_processor.log`
- **Thermal Logs**: Separate thermal event tracking
- **Error Details**: Full stack traces for debugging
- **Progress Metrics**: Detailed timing and performance data

## Performance Optimization

### GPU Optimization
- **Dynamic Batching**: Adjusts batch size based on GPU memory
- **Sequential Processing**: Uses sequential model processing for smaller GPUs
- **Memory Monitoring**: Tracks VRAM usage to prevent OOM errors
- **Thermal Management**: Prevents thermal throttling through proactive cooling

### Processing Efficiency
- **Checkpoint Overhead**: Minimal impact on processing speed
- **Batch Processing**: Optimized batch sizes for your hardware
- **Recovery Speed**: Fast session loading and resumption
- **Resource Management**: Efficient memory and GPU utilization

## Comparison with Standard Processor

| Feature | Standard Processor | Robust Processor |
|---------|-------------------|------------------|
| Progress Saving | ❌ | ✅ Every N papers |
| Thermal Protection | ❌ | ✅ Real-time monitoring |
| Session Recovery | ❌ | ✅ Automatic resume |
| GPU Memory Monitoring | Basic | ✅ Advanced |
| Error Recovery | Limited | ✅ Comprehensive |
| Interruption Handling | ❌ | ✅ Graceful shutdown |
| Thermal Events Log | ❌ | ✅ Detailed tracking |
| Status Monitoring | Basic | ✅ Real-time status |

## Best Practices

### For Long Processing Sessions
1. **Automatic Checkpoints**: Progress saved after each paper by default
2. **Monitor Temperature**: Check thermal status before starting large batches
3. **Conservative Limits**: Use slightly lower temperature thresholds for stability
4. **Regular Monitoring**: Periodically check session status during long runs

### For System Stability
1. **Temperature Limits**: Keep GPU below 80°C for longevity
2. **Batch Sizing**: Start with smaller batches and increase gradually
3. **Memory Management**: Monitor VRAM usage, especially with large models
4. **Cooling Periods**: Allow cooling breaks for sustained processing

### For Recovery
1. **Session Files**: Keep session files safe and backed up
2. **Resume Carefully**: Verify session status before resuming
3. **Clean Restarts**: Clear sessions when changing parameters significantly
4. **Log Analysis**: Review logs to understand interruption causes

## Troubleshooting

### Common Issues

**GPU Temperature Too High**
```bash
# Check current thermal status
python robust_llm_processor.py --thermal-status

# Lower temperature threshold
python robust_llm_processor.py --max-temp 75 --cooling-temp 65
```

**Session Won't Resume**
```bash
# Check session status
python robust_llm_processor.py --session-status

# Clear and restart if corrupted
python robust_llm_processor.py --clear-session
```

**Out of Memory Errors**
```bash
# Reduce batch size
python robust_llm_processor.py --batch-size 2

# Less frequent checkpoints to reduce overhead
python robust_llm_processor.py --checkpoint-interval 5
```

### Log Analysis
- Check `robust_llm_processor.log` for detailed error information
- Thermal events are logged with timestamps and temperatures
- Memory usage is tracked throughout processing
- Performance metrics help optimize batch sizes

## Integration with Existing Pipeline

The robust processor is a drop-in replacement for `run_llm_processing.py` with additional features:

- **Same Database**: Uses the same database and paper collection
- **Same Models**: Works with existing gemma2:9b and qwen2.5:14b models
- **Same Output**: Produces identical intervention extraction results
- **Enhanced Reliability**: Adds safety and recovery features

You can switch between processors as needed:
- Use standard processor for quick tests
- Use robust processor for production runs
- Resume robust sessions with either processor

## Future Enhancements

Potential future improvements:
- **Multi-GPU Support**: Distribute processing across multiple GPUs
- **Cloud Integration**: Support for cloud-based processing
- **Advanced Scheduling**: Intelligent scheduling based on system load
- **Predictive Cooling**: Machine learning-based thermal prediction
- **Resource Optimization**: Dynamic resource allocation based on paper complexity