random_seed = 1234
root = '/home/kotlin/DeepLearning/Whistler/UrbanSound8K'
output_dir = './runs'
device = 'cuda:0'

pin_memory = False
cudnn_benchmark = True
cudnn_deterministic = True
cudnn_enabled = True

hidden_dim = 128
batch = 32
classes = 10
epochs = 80
sample_rate = 22050

dataloader = dict(
    dataset=root,
    batch_size=batch,
    is_shuffle=True,
    num_workers=16,
    drop_last=True,
    sr=sample_rate
)

is_finetune = False
finetune_checkpoint = output_dir

model = dict(
    batch_size=batch,
    d_model=hidden_dim,
    num_classes=classes,
    nhead=8,
    dropout=0.1,
    dim_feedforward=hidden_dim * 2,
    normalize_before=False,
    backbone=dict(
        backbone='resnet18',
        num_feature_levels=1,
        train_backbone=True,
        dilation=False,
        position_embedding='sine',  # 'sine' or 'learned'
        masks=False,
        hidden_dim=hidden_dim,
    ),
    rnn=dict(
        input_size=hidden_dim // 2,
        hidden_size=hidden_dim,
        layers=1,
        dropout=0.1,
        bilstm=False,
    )
)

train = dict(
    learning_rate=1e-5,
    max_lr=1.24e-5,
    epochs=epochs,
    train_step=10,
    batch_size=batch,
    num_classes=classes,
    save_model=output_dir
)

test = dict(
    model_dir=finetune_checkpoint + '/best_model.pt',
    step=10,
    batch_size=batch,
    num_classes=classes,
)