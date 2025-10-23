# Google Colab Setup Guide

This guide will help you train your currency prediction models using Google Colab's free GPU resources.

## Quick Start

1. **Upload to Google Colab**
   - Open [Google Colab](https://colab.research.google.com/)
   - Click `File > Upload notebook`
   - Upload `train_model_colab.ipynb`

2. **Enable GPU** (Important!)
   - Click `Runtime > Change runtime type`
   - Set `Hardware accelerator` to **GPU**
   - Click `Save`

3. **Run the notebook**
   - Follow the cells in order
   - Update the repository URL in Cell 3
   - Choose single currency or batch training

## Features

### Automatic Google Drive Integration
- Models are automatically saved to your Google Drive
- Location: `/MyDrive/CurrencyCapPortal/models/`
- Persists even after Colab session ends
- Includes training logs with metrics and timestamps

### GPU Acceleration
- Free T4 GPU (15x faster than CPU for LSTM training)
- Automatic GPU detection
- Colab Pro offers better GPUs (V100, A100)

### Progress Tracking
- Real-time training progress
- Success/failure summary for batch training
- Training metrics saved to JSON logs

## Step-by-Step Instructions

### Step 1: Prepare Your Repository

Before using Colab, make sure your repository is accessible:

**Option A: Public GitHub Repository (Recommended)**
```bash
# Push your code to GitHub
git add .
git commit -m "Prepare for Colab training"
git push
```

Then update Cell 3 in the notebook with your GitHub URL:
```python
!git clone https://github.com/YOUR_USERNAME/CurrencyCapPortal.git
```

**Option B: Upload Files Manually**
- Skip Cell 3 (clone repository)
- Use Colab's file upload feature to upload:
  - `ml/` directory
  - `api/latest.json`
  - `requirements.txt`
  - Historical data files

### Step 2: Configure Training

**For Single Currency Training (Cell 6):**
```python
CURRENCY = 'usd'  # Change to your desired currency
EPOCHS = 100      # Adjust number of epochs (50-200 typical)
```

**For Batch Training (Cell 7):**
```python
EPOCHS = 100  # Epochs per currency
```

### Step 3: Monitor Training

The notebook will show:
- GPU status and availability
- Training progress for each currency
- Loss metrics
- Success/failure summary
- Estimated time remaining (for batch mode)

### Step 4: Access Trained Models

**From Google Drive:**
1. Open Google Drive
2. Navigate to `/MyDrive/CurrencyCapPortal/models/`
3. Download models as needed

**Direct Download from Colab:**
Run Cell 9 to download all models as a zip file

## Resource Limits

### Free Tier
- **GPU Time:** Limited hours per day (resets daily)
- **Session Duration:** 12 hours maximum
- **Idle Timeout:** 90 minutes
- **RAM:** ~12GB
- **Disk:** ~100GB

### Tips for Free Tier
- Train models in batches if you have many currencies
- Keep browser tab active to prevent idle timeout
- Monitor GPU quota (shown in Colab interface)

### Colab Pro ($10/month)
- 24-hour sessions
- More GPU time
- Better GPUs (V100, A100)
- Priority access
- Background execution

## Troubleshooting

### GPU Not Available
**Issue:** Cell 1 shows "No GPU detected"
**Solution:**
1. `Runtime > Change runtime type`
2. Select `GPU` under Hardware accelerator
3. Click `Save`
4. Re-run Cell 1

### Out of Memory Error
**Issue:** Training crashes with OOM error
**Solutions:**
- Reduce batch size in your `CurrencyPredictor` class
- Train fewer currencies at once
- Restart runtime: `Runtime > Restart runtime`
- Clear output: `Edit > Clear all outputs`

### Session Disconnected
**Issue:** "Reconnecting..." message appears
**Solutions:**
- Keep browser tab active
- Disable browser sleep/hibernation
- Use Colab Pro for longer sessions
- Models are safe in Google Drive even if disconnected

### Module Not Found Error
**Issue:** `ModuleNotFoundError: No module named 'ml'`
**Solutions:**
- Make sure Cell 3 (clone repo) ran successfully
- Check that you're in the correct directory: `%cd CurrencyCapPortal`
- Verify `ml/` directory exists: `!ls -la`

### Google Drive Permission Error
**Issue:** Cannot write to Drive
**Solution:**
- Re-run Cell 2 (Mount Google Drive)
- Click the authorization link
- Grant permission to access Drive

## Best Practices

### Before Training
- [ ] Enable GPU runtime
- [ ] Mount Google Drive (Cell 2)
- [ ] Verify repository is cloned/uploaded
- [ ] Check `api/latest.json` exists
- [ ] Ensure historical data files are present

### During Training
- [ ] Keep browser tab active
- [ ] Monitor GPU usage
- [ ] Watch for errors in output
- [ ] Note any failed currencies

### After Training
- [ ] Verify models saved to Drive
- [ ] Check training logs
- [ ] Download models if needed
- [ ] Review metrics for quality

## Advanced Usage

### Custom Hyperparameters

Edit the `CurrencyPredictor` class before training:
```python
# Add a cell before Cell 6
from ml.predictor import CurrencyPredictor

# Monkey-patch for custom parameters
original_init = CurrencyPredictor.__init__

def custom_init(self, currency_code, lookback=60, epochs=100):
    original_init(self, currency_code)
    self.lookback = lookback  # Custom lookback period

CurrencyPredictor.__init__ = custom_init
```

### Training Specific Currencies

To train only specific currencies instead of all:
```python
# Custom list
currencies_to_train = ['usd', 'eur', 'gbp', 'jpy']

for currency in currencies_to_train:
    train_single_currency(currency, epochs=100, save_to_drive=True)
```

### Resuming Failed Training

If some currencies failed:
```python
# List failed currencies from summary
failed_currencies = ['cad', 'aud']

for currency in failed_currencies:
    train_single_currency(currency, epochs=150, save_to_drive=True)
```

## Performance Benchmarks

Approximate training times (with GPU):

| Currencies | Epochs | Time Estimate |
|-----------|--------|---------------|
| 1 | 100 | 2-5 minutes |
| 10 | 100 | 20-50 minutes |
| 50 | 100 | 2-4 hours |
| 100+ | 100 | 4-8 hours |

*Times vary based on data size and model complexity*

## Cost Comparison

| Option | Cost | GPU | Session | Best For |
|--------|------|-----|---------|----------|
| Colab Free | $0 | T4 | 12h | Testing, small batches |
| Colab Pro | $10/mo | V100/A100 | 24h | Regular training |
| Local GPU | Hardware cost | Your GPU | Unlimited | Frequent training |

## Next Steps

After training:
1. Copy models from Drive to your production server
2. Update your API to use the trained models
3. Test predictions with `ml/predictor.py`
4. Set up automated retraining schedule

## Support

- **Colab Issues:** [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- **GPU Quota:** Check [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)
- **General Help:** Open an issue in your repository
