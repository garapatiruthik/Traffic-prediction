@echo off
cd /d "C:\Users\p\Documents\Traffic prediction"
"C:\Users\p\AppData\Local\Programs\Python\Python311\python.exe" "C:\Users\p\Documents\Traffic prediction\step5_mamba_training.py" > step5_output.log 2>&1
echo DONE >> step5_output.log
