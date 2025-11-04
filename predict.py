import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import torch.nn as nn
import torch.nn.functional as F


# 假设你的 SimpleCNN 定义在同一个文件中，或者从其他模块导入
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict_image(image_path, model_path, num_classes=10):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 图像预处理（必须与训练时相同）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载和预处理图像
    image = Image.open(image_path)
    # 如果图像是灰度图，转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    image_tensor = image_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    # 获取预测结果
    predicted_class = predicted.item()
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型预测图像')
    parser.add_argument('--image_path', type=str, required=True, help='要预测的图像路径')
    parser.add_argument('--model_path', type=str, default='model.pth', help='模型路径')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    args = parser.parse_args()
    
    # CIFAR-10 的类别名称
    if args.num_classes == 10:
        class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    else:
        class_names = [f'类别{i}' for i in range(args.num_classes)]
    
    # 进行预测
    predicted_class, confidence = predict_image(
        args.image_path, 
        args.model_path, 
        args.num_classes
    )
    
    print(f"预测结果: {class_names[predicted_class]}")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")

if __name__ == '__main__':
    main()