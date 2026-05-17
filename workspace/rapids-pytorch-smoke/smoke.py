import torch
import cudf
from cuml.ensemble import RandomForestClassifier
from cuml.model_selection import train_test_split
import numpy as np

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("PyTorch CUDA matmul OK:", y.shape)

df = cudf.DataFrame({
    "a": np.random.rand(1000),
    "b": np.random.rand(1000),
})
df["target"] = (df["a"] + df["b"] > 1).astype("int32")
print("cuDF OK:")
print(df.head())

X = df[["a", "b"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("cuML RandomForest score:", float(score))
print("Baseline RAPIDS + PyTorch test passed.")
