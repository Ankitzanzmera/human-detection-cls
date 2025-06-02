
import os
import numpy as np
from tqdm import tqdm
import os.path as osp
from torch.utils.data import DataLoader
from torchmetrics.classification import (BinaryAveragePrecision,
                                         BinaryAUROC,
                                         BinaryF1Score,
                                         BinaryRecall,
                                         BinaryPrecision,
                                         BinaryAccuracy)

from human_detection_cls.config import *
from human_detection_cls.data_preparation import make_csv
from human_detection_cls.utils import plot_label_counts
from human_detection_cls.logger import get_training_log_config
from human_detection_cls.datasets import BinarDataset
from human_detection_cls.augmentation import get_transforms
from human_detection_cls.models import DensnetModel
from human_detection_cls.callback import EarlyStopping
from human_detection_cls.utils import plot_class_wise_ap_auroc, plot_metrics


if __name__ == "__main__":

    os.makedirs(ARTIFACT_DIR_PATH, exist_ok=True)
    
    logger = get_training_log_config(ARTIFACT_DIR_PATH)
    logger.info(f"ARTIFACT_DIR_PATH path : {ARTIFACT_DIR_PATH}")
    
    train_csv, valid_csv = make_csv(IMAGE_DIR, LABEL_MAP)
    logger.info(f"Train Label Counts : {train_csv['labels'].value_counts().to_dict()}")
    logger.info(f"Train Label Counts : {valid_csv['labels'].value_counts().to_dict()}")
    plot_label_counts(train_csv, ARTIFACT_DIR_PATH, flag = "train")
    plot_label_counts(valid_csv, ARTIFACT_DIR_PATH, flag = "valid")
    
    train_transform, valid_transform = get_transforms(size = IMAGE_SIZE)
    train_dataset = BinarDataset(train_csv, LABEL_MAP, train_transform)
    valid_dataset = BinarDataset(valid_csv, LABEL_MAP, valid_transform)
    
    train_dataloader = DataLoader(train_dataset, shuffle = True, drop_last = True, pin_memory = True, batch_size = 8)
    valid_dataloader = DataLoader(valid_dataset, shuffle = True, drop_last = True, pin_memory = True, batch_size = 8)
    
    model = DensnetModel(num_classes=1, pretrained=True).to(DEVICE)
    early_stopping = EarlyStopping(patience=10, verbose=True, save_path=osp.join(ARTIFACT_DIR_PATH, "best_model.pth"))
    optimizer = get_model_parameters(model)
    
    epochwise_train_acc, epochwise_val_acc = [], []
    epochwise_train_loss, epochwise_val_loss = [], []
    classwise_ap_history, classwise_auroc_history, classwise_precision_history, classwise_recall_history, classwise_f1_score_history = [], [], [], [], []
    
    binary_label_AP = BinaryAveragePrecision(thresholds = None)
    binary_label_AUROC = BinaryAUROC(thresholds = None)
    binary_label_precision = BinaryPrecision()
    binary_label_recall = BinaryRecall()
    binary_label_f1_score = BinaryF1Score()
    binary_label_train_accuracy = BinaryAccuracy()
    binary_label_valid_accuracy = BinaryAccuracy()
    
    
    for epoch in range(1, NUM_EPOCHS+1):
        
        model.train()
        train_batch_loss = []
        for batch in tqdm(train_dataloader):
            image = batch['image'].to(DEVICE)
            y_true = batch['label'].unsqueeze(1).to(DEVICE)
            
            y_pred = model(image)
            loss = CRITERION(y_true, y_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_batch_loss.append(loss.item())
            binary_label_train_accuracy.update(y_pred.detach().cpu(), y_true.detach().cpu())
            
        train_acc = binary_label_train_accuracy.compute().item()
        epochwise_train_acc.append(train_acc)
        train_loss = np.mean(train_batch_loss)
        epochwise_train_loss.append(train_loss)
            
        
        valid_batch_loss = []
        model.eval()
        for batch in tqdm(valid_dataloader):
            image = batch['image'].to(DEVICE)
            y_true = batch['label'].unsqueeze(1).to(DEVICE)
                
            with torch.no_grad():
                y_pred = model(image)
                loss = CRITERION(y_pred, y_true)
                
            valid_batch_loss.append(loss.item())
            binary_label_valid_accuracy.update(y_pred.detach().cpu(), y_true.detach().cpu())
            binary_label_AP.update(y_pred.float().detach().cpu(), y_true.long().detach().cpu())
            binary_label_AUROC.update(y_pred.float().detach().cpu(), y_true.long().detach().cpu())
            binary_label_precision.update(y_pred.float().detach().cpu(), y_true.long().detach().cpu())
            binary_label_recall.update(y_pred.float().detach().cpu(), y_true.long().detach().cpu())
            binary_label_f1_score.update(y_pred.float().detach().cpu(), y_true.long().detach().cpu())    
        
        val_acc = binary_label_valid_accuracy.compute().item()
        epochwise_val_acc.append(val_acc)
        val_loss = np.mean(valid_batch_loss)
        epochwise_val_loss.append(val_loss)
        
        ap    = binary_label_AP.compute().mean().item()
        auroc = binary_label_AUROC.compute().mean().item()
        precision = binary_label_precision.compute().mean().item()
        recall = binary_label_recall.compute().mean().item()
        f1_score = binary_label_f1_score.compute().mean().item()
        
        classwise_ap_history.append(ap)
        classwise_auroc_history.append(auroc)
        classwise_precision_history.append(precision)
        classwise_recall_history.append(recall)
        classwise_f1_score_history.append(f1_score)
        
        epoch_print = f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, mAP: {ap:.3f}, mAUROC: {auroc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1_score:.3f}"
        print(epoch_print)  
        logger.info(epoch_print)

        plot_metrics(epochwise_train_loss, epochwise_val_loss, epochwise_train_acc, epochwise_val_acc, epoch, ARTIFACT_DIR_PATH)
        plot_class_wise_ap_auroc(classwise_ap_history, classwise_auroc_history, classwise_precision_history, classwise_recall_history, classwise_f1_score_history, epoch, ARTIFACT_DIR_PATH)
        
        early_stopping(val_loss, model, logger)
        if early_stopping.early_stop:
            print("Model is Not Improving, Therefore Early Stopping callback is Triggered....")
            logger.info("Model is Not Improving, Therefore Early Stopping callback is Triggered....")
            break
        
        binary_label_AP.reset()
        binary_label_AUROC.reset()
        binary_label_precision.reset()
        binary_label_recall.reset()
        binary_label_f1_score.reset()
        binary_label_train_accuracy.reset()
        binary_label_valid_accuracy.reset()
        print("--" * 75)
            
    
    
    
    
    
    

    
    
    

    
    
    
    
    