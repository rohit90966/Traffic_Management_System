import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import tempfile
import streamlit as st

# ================== Define CBAM Components ==================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# =============== Define CBAMResNet18 Model ===================
import torchvision.models as models

class CBAMResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(CBAMResNet18, self).__init__()
        self.base = models.resnet18(pretrained=False)  # Change to False if no internet or custom weights
        self.base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # Hook functions to apply CBAM on the outputs of layers
        self.base.layer1[0].register_forward_hook(lambda m, i, o: self.cbam1(o))
        self.base.layer2[0].register_forward_hook(lambda m, i, o: self.cbam2(o))
        self.base.layer3[0].register_forward_hook(lambda m, i, o: self.cbam3(o))
        self.base.layer4[0].register_forward_hook(lambda m, i, o: self.cbam4(o))

        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# ======================= Load model ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CBAMResNet18(num_classes=2)
model.load_state_dict(torch.load("models/cbam_resnet18_weights.pth", map_location=device))
model.to(device)
model.eval()

# =================== Preprocess audio ========================
def audio_to_tensor(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot spectrogram to image buffer
    plt.figure(figsize=(1.28, 1.28), dpi=100)  # 128x128 px
    plt.axis('off')
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)

    img = Image.open(buf).convert('L')  # grayscale image

    # Transform as per training
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)

# ===================== Prediction ============================
def predict_siren(file_path):
    tensor = audio_to_tensor(file_path)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        # Assuming class 0 = siren, class 1 = no siren (adjust if needed)
        siren_score = probs[0][0].item()
    return siren_score

# ===================== Streamlit UI ==========================
st.title("🚨 Siren Sound Detection (PyTorch CBAM Model)")

option = st.radio("Choose an option:", ("Upload Audio File", "Record Audio (Not supported here)"))

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload audio file (wav or mp3)", type=['wav', 'mp3'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.audio(tmp_file_path)

        try:
            score = predict_siren(tmp_file_path)
            if score < 0.5:
                st.success(f"🔇 No Siren Detected. Confidence: {score:.4f}")
            else:
                st.info(f"🚨 SIREN Detected! Confidence: {score:.4f}")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

elif option == "Record Audio (Not supported here)":
    st.info("Audio recording is not supported in this app yet. Please upload an audio file.")
