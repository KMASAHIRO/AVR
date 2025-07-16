from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
import matplotlib.pyplot as plt

# === ログファイルのパス ===
log_path = "/home/ach17616qc/tensorboard_logs/real_exp/Real_exp_param_1_1/0707-153722/events.out.tfevents.1751870242.hnode044.291534.0"

# === イベント読み込み ===
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# === スカラータグを分類 ===
train_loss_tags = [tag for tag in ea.Tags()['scalars'] if tag.startswith('train_loss/') and tag != 'train_loss']
test_loss_tags = [tag for tag in ea.Tags()['scalars'] if tag.startswith('test_loss/')]

# === ステップごとのスカラー加算 ===
def accumulate_tags(tags):
    acc = defaultdict(float)
    for tag in tags:
        for event in ea.Scalars(tag):
            acc[event.step] += event.value
    return acc

train_loss_sum = accumulate_tags(train_loss_tags)
test_loss_sum = accumulate_tags(test_loss_tags)

# === ステップ順に並べる ===
train_steps, train_values = zip(*sorted(train_loss_sum.items()))
test_steps, test_values = zip(*sorted(test_loss_sum.items()))

# === ステップをエポックに変換 ===
first_step = min(train_steps)
train_epochs = [step / first_step for step in train_steps]
test_epochs = [step / first_step for step in test_steps]

# === グラフ描画と保存 ===
plt.figure(figsize=(10, 5))
plt.plot(train_epochs, train_values, label='Train Loss', color='blue')
plt.plot(test_epochs, test_values, label='Test Loss', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Test Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === PNGファイル出力 ===
plt.savefig("loss_plot_by_epoch.png")
plt.close()
