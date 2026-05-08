nhớ tải môi trường: pip install torch torchvision tqdm opencv-python pillow matplotlib scikit-image

Bước 1: chạy python dataset.py 
 màn hình in ra được hình dáng của tensor 
 (ví dụ:Image batch shape: torch.Size([4, 1, 256, 256])
        Mask batch shape: torch.Size([4, 1, 256, 256]) hay 512 gì đó.
        Prompt batch shape: torch.Size([4, 1, 256, 256])
        Giá trị đỉnh của một Prompt: 1.00
 )

 Bước 2: chạy python train.py