import numpy as np
import pytest

from psyinsight.dl import TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")


def test_mlp_classifier_trains_and_predicts():
    from torch.utils.data import DataLoader

    from psyinsight.dl import MLPClassifier, TabularDataset, Trainer

    X = np.random.rand(120, 4).astype("float32")
    y = (X[:, 0] + X[:, 1] > 1).astype("float32")

    dataset = TabularDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MLPClassifier(input_dim=4, n_classes=2)
    trainer = Trainer(model, task="binary", lr=1e-2)
    history = trainer.fit(loader, loader, epochs=2, patience=1, verbose=False)

    assert len(history["train_loss"]) >= 1
    preds = trainer.predict_proba(loader)
    assert preds.shape[0] == len(X)


def test_sequence_dataset_and_lstm():
    from torch.utils.data import DataLoader

    from psyinsight.dl import LSTMTextClassifier, SequenceDataset, Trainer, build_vocab, texts_to_sequences

    texts = ["good day", "bad day", "great news", "terrible news"] * 5
    labels = [1, 0, 1, 0] * 5
    vocab = build_vocab(texts)
    sequences = texts_to_sequences(texts, vocab)

    dataset = SequenceDataset(sequences, labels, max_len=5)
    loader = DataLoader(dataset, batch_size=8)

    model = LSTMTextClassifier(vocab_size=len(vocab), n_classes=2)
    trainer = Trainer(model, task="binary", lr=1e-2)
    trainer.fit(loader, epochs=1, verbose=False)
    preds = trainer.predict(loader)
    assert preds.shape[0] == len(texts)
