# OCR_pipline
## Combine CFAFT (text detection) and SCATTER (text recognition)
- Download CRAFT weights: [General Model](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view) and [Link refiner](https://drive.google.com/file/d/1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO/view)
- Put CRAFT weights in: craft_text_detector/weights
- Download SCATTER weights: [SCATTER](https://drive.google.com/drive/u/1/folders/1niuPM6otpSQFSai8Ft2bO0lhdqEjE96Z)
- Put SCATTER weight (scatter-case-sensitive.pth) in: scatter_scatter_text_recognizer/weights. More details about this model can be found [here](https://github.com/phantrdat/cvpr20-scatter-text-recognizer)

## Usage
Check in main.py for usage.

There is two versions: 
- Original: First use CRAFT to detection, then use SCATTER for recognition
- Enhanced: First split input into smaller patches (for detection better), then detection and recognition
