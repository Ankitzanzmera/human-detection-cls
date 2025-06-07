# Human Detection Classification Project

### This project uses a dataset for binary classification of images into two categories: `human` and `no_human`.

## Installations

1. **Clone this repo using following command:**

   ```
   git clone https://github.com/Ankitzanzmera/human-detection-cls
   ```

2. **Create Virtual environment using folllowing commands:**

   ```
   conda create -n human-detection-cls python=3.9 -y
   ```

3. **Install All dependencies from requirements file**:
   
   ``` 
   pip install -r rrequirments.txt
   ```
   

## Download Dataset
Follow the steps below to download and prepare the dataset:

1. **Download the dataset** from Kaggle:  
   [🔗 Human Detection Dataset](https://www.kaggle.com/datasets/constantinwerner/human-detection-dataset/data)

2. **Extract the zip file** into the `data/` directory of this project.

3. **Rename the extracted folder** by replacing spaces with underscores:  
   - Rename `Human Detection Dataset` to `human_detection_dataset`

4. **Rename the class folders** for clarity:
   - `0` → `no_human`
   - `1` → `human`
5. 📁 **Final Directory Structure** :
    - After completing the steps above, your project directory should look like this:
    ```
    └── human_detection_dataset/
        ├── human
        └── no_human
    ```
---

## Training
   -  **Run train.py using following command:** 
   
      ```
      python train.py 
      ```

   - If everything is correct then this train.py will create one artifact directon where there will subdirectory of current exp. 
   - In this subdirectory you will find all metrics plots and also best model.
   - This code also create folder called saved model where only best model of current experiement will stored.

## Inference (Local)
- for model inference you have to run inference.py with some argument which shown below :

   ```
    python inference.py --img_path <path> --device --save
    ```
  
   - inplace of PATH you have to replce actual path of image.
   - if --device given then it will use cuda for inference if available, otherwise it will use CPU
   - if --save given then it will create preds directory and will save output of inference in it.

## Inference (Web based):
   - For web based inference i have used streamlit. in which use have to upload image and need to click on predict button.
   - after clicking predict button it will automatically show output with heatmaps (grad-cam)and probability.

      ```
      streamlit run app.py 
      ```
   - run above command to run app.py


## Docker Deployment
Follow the step for deploy my project in docker

   -  go into project directory and write following command:
      
      ``` 
      docker build -t <IMAGE NAME YOU WANT TO GIVE> .
      ```

   - it will create docker image and will install all the dependencies needs to run project.

   - then we need to create docker container for that use following command : 

      ``` 
      docker run -it <GIVE CONTAINER NAME> -p 8501:8501 <IMAG NAME>
      ```

   - Above command will create and run the container.

   - Click on link shown in terminal to go on browser. or else start any browser and write following URL:
   
       ``` 
       https://localhost:8501 
       ```