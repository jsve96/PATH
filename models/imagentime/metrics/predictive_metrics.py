import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.utils import extract_time


class PredictorGRU(nn.Module):
    """
    Post-hoc GRU predictor used for TimeGAN predictive score.

    Input:
        x: shape (batch, seq_len - 1, dim - 1)

    Output:
        y_hat: shape (batch, seq_len - 1, 1)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor of shape (batch, time, input_dim)
            lengths: Optional Tensor/List of valid sequence lengths.

        Returns:
            y_hat: Tensor of shape (batch, time, 1)
        """

        if lengths is not None:
            lengths_cpu = lengths.detach().cpu()

            packed_x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )

            packed_out, _ = self.gru(packed_x)

            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
                total_length=x.size(1),
            )
        else:
            out, _ = self.gru(x)

        y_hat_logit = self.fc(out)
        y_hat = self.sigmoid(y_hat_logit)

        return y_hat


def _make_predictor_batch(data, times, indices, max_seq_len, dim, device):
    """
    Builds one batch for the post-hoc predictor.

    For each sequence:
        X_t = first dim - 1 features at time t
        Y_t = last feature at time t + 1

    Args:
        data: array/list of shape (N, seq_len, dim)
        times: sequence lengths
        indices: selected sample indices
        max_seq_len: maximum sequence length
        dim: feature dimension
        device: torch device

    Returns:
        X: shape (batch, max_seq_len - 1, dim - 1)
        Y: shape (batch, max_seq_len - 1, 1)
        lengths: shape (batch,)
        mask: shape (batch, max_seq_len - 1, 1)
    """

    batch_size = len(indices)
    pred_seq_len = max_seq_len - 1

    X = np.zeros((batch_size, pred_seq_len, dim - 1), dtype=np.float32)
    Y = np.zeros((batch_size, pred_seq_len, 1), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for b, idx in enumerate(indices):
        seq = np.asarray(data[idx], dtype=np.float32)

        # Number of valid one-step prediction targets.
        curr_len = int(times[idx]) - 1
        curr_len = max(1, min(curr_len, pred_seq_len, seq.shape[0] - 1))

        X[b, :curr_len, :] = seq[:curr_len, : dim - 1]
        Y[b, :curr_len, 0] = seq[1 : curr_len + 1, dim - 1]

        lengths[b] = curr_len

    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)
    lengths = torch.tensor(lengths, device=device)

    time_grid = torch.arange(pred_seq_len, device=device).unsqueeze(0)
    mask = time_grid < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    return X, Y, lengths, mask


def predictive_score_metrics(
    ori_data,
    generated_data,
    iterations=5000,
    batch_size=128,
    device=None,
):
    """
    PyTorch version of TimeGAN predictive score.

    Trains a post-hoc GRU on synthetic/generated data to predict the
    next-step value of the last feature, then evaluates MAE on original data.

    Args:
        ori_data:
            Original data of shape (N, seq_len, dim).

        generated_data:
            Generated synthetic data of shape (N, seq_len, dim).

        iterations:
            Number of predictor training iterations.

        batch_size:
            Mini-batch size for predictor training.

        device:
            Optional torch device. If None, uses CUDA if available.

    Returns:
        predictive_score:
            Mean absolute error on original data.
    """

    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)

    no, seq_len, dim = ori_data.shape

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Sequence lengths.
    ori_time, ori_max_seq_len = extract_time(ori_data)

    # Important: this should use generated_data, not ori_data.
    generated_time, generated_max_seq_len = extract_time(generated_data)

    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    hidden_dim = int(dim / 2)

    predictor = PredictorGRU(
        input_dim=dim - 1,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = optim.Adam(predictor.parameters())

    predictor.train()

    # ------------------------------------------------------------
    # Train predictor on generated synthetic data.
    # ------------------------------------------------------------
    for _ in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_mb, Y_mb, T_mb, mask_mb = _make_predictor_batch(
            generated_data,
            generated_time,
            train_idx,
            max_seq_len,
            dim,
            device,
        )

        optimizer.zero_grad()

        Y_pred = predictor(X_mb, T_mb)

        abs_error = torch.abs(Y_mb - Y_pred)
        loss = (abs_error * mask_mb).sum() / mask_mb.sum().clamp_min(1.0)

        loss.backward()
        optimizer.step()

    # ------------------------------------------------------------
    # Evaluate predictor on original data.
    # ------------------------------------------------------------
    predictor.eval()

    with torch.no_grad():
        idx = np.random.permutation(len(ori_data))
        test_idx = idx[:no]

        X_mb, Y_mb, T_mb, mask_mb = _make_predictor_batch(
            ori_data,
            ori_time,
            test_idx,
            max_seq_len,
            dim,
            device,
        )

        Y_pred = predictor(X_mb, T_mb)

        abs_error = torch.abs(Y_mb - Y_pred) * mask_mb

        # Match the original TimeGAN logic:
        # compute MAE per sequence, then average over sequences.
        per_sequence_mae = abs_error.squeeze(-1).sum(dim=1) / T_mb.float().clamp_min(1.0)
        predictive_score = per_sequence_mae.mean().item()

    return predictive_score




# """Time-series Generative Adversarial Networks (TimeGAN) Codebase.

# Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
# "Time-series Generative Adversarial Networks," 
# Neural Information Processing Systems (NeurIPS), 2019.

# Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

# Last updated Date: April 24th 2020
# Code author: Jinsung Yoon (jsyoon0823@gmail.com)

# -----------------------------

# predictive_metrics.py

# Note: Use Post-hoc RNN to predict one-step ahead (last feature)
# """

# # Necessary Packages
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import mean_absolute_error
# from utils.utils import extract_time

# tf.compat.v1.disable_eager_execution()


# def predictive_score_metrics(ori_data, generated_data):
#     """Report the performance of Post-hoc RNN one-step ahead prediction.

#     Args:
#       - ori_data: original data
#       - generated_data: generated synthetic data

#     Returns:
#       - predictive_score: MAE of the predictions on the original data
#     """
#     # Initialization on the Graph
#     tf.compat.v1.reset_default_graph()

#     # Basic Parameters
#     no, seq_len, dim = np.asarray(ori_data).shape

#     # Set maximum sequence length and each sequence length
#     ori_time, ori_max_seq_len = extract_time(ori_data)
#     generated_time, generated_max_seq_len = extract_time(ori_data)
#     max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

#     ## Builde a post-hoc RNN predictive network
#     # Network parameters
#     hidden_dim = int(dim / 2)
#     iterations = 5000
#     batch_size = 128

#     # Input place holders
#     X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len - 1, dim - 1], name="myinput_x")
#     T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")
#     Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len - 1, 1], name="myinput_y")

#     # Predictor function
#     def predictor(x, t):
#         """Simple predictor function.

#         Args:
#           - x: time-series data
#           - t: time information

#         Returns:
#           - y_hat: prediction
#           - p_vars: predictor variables
#         """
#         with tf.compat.v1.variable_scope("predictor", reuse=tf.compat.v1.AUTO_REUSE) as vs:
#             p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='p_cell')
#             p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
#             y_hat_logit = tf.compat.v1.layers.dense(p_outputs, 1, activation=None)
#             y_hat = tf.nn.sigmoid(y_hat_logit)
#             p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]

#         return y_hat, p_vars

#     y_pred, p_vars = predictor(X, T)
#     # Loss for the predictor
#     p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
#     # optimizer
#     p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)

#     ## Training
#     # Session start
#     sess = tf.compat.v1.Session()
#     sess.run(tf.compat.v1.global_variables_initializer())

#     # Training using Synthetic data
#     for itt in range(iterations):
#         # Set mini-batch
#         idx = np.random.permutation(len(generated_data))
#         train_idx = idx[:batch_size]

#         X_mb = list(generated_data[i][:-1, :(dim - 1)] for i in train_idx)
#         T_mb = list(generated_time[i] - 1 for i in train_idx)
#         Y_mb = list(
#             np.reshape(generated_data[i][1:, (dim - 1)], [len(generated_data[i][1:, (dim - 1)]), 1]) for i in train_idx)

#         # Train predictor
#         _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})

#         ## Test the trained model on the original data
#     idx = np.random.permutation(len(ori_data))
#     train_idx = idx[:no]

#     X_mb = list(ori_data[i][:-1, :(dim - 1)] for i in train_idx)
#     T_mb = list(ori_time[i] - 1 for i in train_idx)
#     Y_mb = list(np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1]) for i in train_idx)

#     # Prediction
#     pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

#     # Compute the performance in terms of MAE
#     MAE_temp = 0
#     for i in range(no):
#         MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])

#     predictive_score = MAE_temp / no

#     return predictive_score
