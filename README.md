# sumoflow-ai-traffic-optimization
AI-Driven Traffic Optimization using YOLO and SUMO - Graduation Project 

## Simulation Outputs
Large output files (fcd.xml, emission.xml, ssm.xml etc.) are available here:
[Google Drive Link](https://drive.google.com/drive/u/0/folders/1ULhxaaJfYKngSDmra5949AVY5koPnaeH)



### Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/ranna-waleed/sumoflow-ai-traffic-optimization.git
cd sumoflow-ai-traffic-optimization
```

**Step 2: Create virtual environment**
```bash
python -m venv sumoflow_env
```

**Step 3: Activate virtual environment**
```bash
# Windows
sumoflow_env\Scripts\activate

# Mac/Linux
source sumoflow_env/bin/activate
```

**Step 4: Install all requirements**
```bash
pip install -r requirements.txt
```

**Step 5: Verify installation**
```bash
python -c "import traci; import cv2; import ultralytics; print('All packages installed successfully!')"
```

### Deactivate Environment When Done
```bash
deactivate
```
