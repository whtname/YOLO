import torch
import numpy as np
from torchvision.transforms import ToTensor
from utils.paths import CHECKPOINTS_DIR

GPU_EFFICIENT_SAM_CHECKPOINT = CHECKPOINTS_DIR / "efficient_sam_s_gpu.jit"
CPU_EFFICIENT_SAM_CHECKPOINT = CHECKPOINTS_DIR / "efficient_sam_s_cpu.jit"


def load(device: str) -> torch.jit.ScriptModule:
    if device == 'cuda':
        model = torch.jit.load(GPU_EFFICIENT_SAM_CHECKPOINT)
    else:
        model = torch.jit.load(CPU_EFFICIENT_SAM_CHECKPOINT)
    model.eval()
    return model

def inference_with_box( 
    image: np.ndarray, 
    box: np.ndarray, 
    model: torch.jit.ScriptModule, 
    device: torch.device 
) -> np.ndarray: 
    
    with torch.no_grad():
        bbox = torch.reshape(torch.tensor(box), [1, 1, 2, 2]).to(device) # 直接将tensor发送到device
        bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2]).to(device) # 直接将tensor发送到device
        
        img_tensor = ToTensor()(image)[None, ...].to(device) 

        predicted_logits, predicted_iou = model( 
            img_tensor, 
            bbox, 
            bbox_labels, 
        ) 
        
        predicted_logits = predicted_logits.cpu() 
        all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy() 
        
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy() 

    max_predicted_iou = -1 
    selected_mask_using_predicted_iou = None 
    for m in range(all_masks.shape[0]): 
        curr_predicted_iou = predicted_iou[m] 
        if ( 
                curr_predicted_iou > max_predicted_iou 
                or selected_mask_using_predicted_iou is None 
        ): 
            max_predicted_iou = curr_predicted_iou 
            selected_mask_using_predicted_iou = all_masks[m] 
            
    return selected_mask_using_predicted_iou

def inference_with_boxes(
    image: np.ndarray,
    xyxy: np.ndarray,
    model: torch.jit.ScriptModule,
    device: torch.device
) -> np.ndarray:
    masks = []
    print(len(xyxy))
    for [x_min, y_min, x_max, y_max] in xyxy:
        box = np.array([[x_min, y_min], [x_max, y_max]])
        mask = inference_with_box(image, box, model, device)
        masks.append(mask)
    return np.array(masks)