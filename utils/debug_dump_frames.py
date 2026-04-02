import os
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms as T

PREDS_CSV = "experiments/FFPP_10/eval/celebdf_v2_test_preds.csv"
OUTDIR = "experiments/FFPP_10/eval/debug_top_confident"
os.makedirs(OUTDIR, exist_ok=True)

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

df = pd.read_csv(PREDS_CSV)

# ----- pick top confident frames -----
K = 40  # total frames to save
K0 = K // 2  # top real (lowest prob)
K1 = K - K0  # top fake (highest prob)

top_real = df[df["label"] == 0].sort_values("prob", ascending=True).head(K0).copy()
top_fake = df[df["label"] == 1].sort_values("prob", ascending=False).head(K1).copy()

sel = pd.concat([top_real, top_fake], ignore_index=True)
sel["rank"] = range(len(sel))

def read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

# ----- save frames -----
for i, row in tqdm(sel.iterrows(), total=len(sel), desc="Saving top-confident frames"):
    path = row["path"]
    label = int(row["label"])
    prob = float(row["prob"])

    # path can be an image or mp4. handle both.
    if str(path).lower().endswith(".mp4"):
        frame = read_first_frame(path)
        if frame is None:
            print("Could not read video:", path)
            continue
        img = Image.fromarray(frame)
    else:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print("Could not open image:", path, "err:", e)
            continue

    x = transform(img)

    # file name with info
    fname = f"{i:03d}_label{label}_p{prob:.4f}.png"
    save_image(x, os.path.join(OUTDIR, fname))

print("Saved", len(sel), "frames to", OUTDIR)


# import os
# import cv2
# import pandas as pd
# from tqdm import tqdm
# from PIL import Image
# from torchvision.utils import save_image
# import torchvision.transforms as T

# CSV = "/scratch/sahil/projects/img_deepfake/datasets/celebdf/split_csv/celeb_df_v2_test.csv"
# OUTDIR = "experiments/FFPP_10/eval/debug_frames"
# os.makedirs(OUTDIR, exist_ok=True)

# df = pd.read_csv(CSV)

# transform = T.Compose([
#     T.Resize((256, 256)),
#     T.ToTensor()
# ])

# N = 32
# N0 = N // 2
# N1 = N - N0

# df0 = df[df["label"] == 0].sample(n=min(N0, (df["label"] == 0).sum()), random_state=42)
# df1 = df[df["label"] == 1].sample(n=min(N1, (df["label"] == 1).sum()), random_state=42)

# sel = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)

# for i in tqdm(range(len(sel)), desc="Saving debug frames (balanced)"):
#     row = sel.iloc[i]
#     video_path = row["path"]
#     label = int(row["label"])

#     cap = cv2.VideoCapture(video_path)
#     ok, frame = cap.read()
#     cap.release()

#     if not ok or frame is None:
#         print("Could not read:", video_path)
#         continue

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(frame)
#     img = transform(img)

#     save_image(img, f"{OUTDIR}/{i:03d}_label{label}.png")

# print("Saved", len(sel), "frames to", OUTDIR)