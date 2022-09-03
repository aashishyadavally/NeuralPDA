import torch
import torch.nn as nn

from utils import load_w2v_model


class SelfAttention(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_heads,
        projection_bias=False,
    ):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.head_dim = embedding_size // num_heads

        # Make sure the number of heads is a factor of the embedding size.
        assert embedding_size % num_heads == 0

        self.k_linear = nn.Linear(self.head_dim, self.head_dim,
                                  bias=projection_bias)
        self.q_linear = nn.Linear(self.head_dim, self.head_dim,
                                  bias=projection_bias)
        self.v_linear = nn.Linear(self.head_dim, self.head_dim,
                                  bias=projection_bias)
        self.fc_out = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, keys, queries, values, mask):
        N = queries.shape[0]
        key_len, query_len, value_len = (keys.shape[1], queries.shape[1],
                                         values.shape[1])

        # Split embedding into individual heads.
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)

        # Pass keys, values, queries through Linear layer.
        keys = self.k_linear(keys)
        queries = self.q_linear(queries)
        values = self.v_linear(queries)

        # Compute energy based on shapes of keys, queries, and values.
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        mask = mask.unsqueeze(1).unsqueeze(2)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        softmax_qk = torch.softmax(energy / (self.embedding_size ** 0.5), dim=3)

        attention = torch.einsum("nhql, nlhd -> nqhd", [softmax_qk, values])\
                         .reshape(N, query_len, self.num_heads * self.head_dim)

        output = self.fc_out(attention)
        return output


class StatementEncoder(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_layers,
        vocab_size,
        max_tokens,
        num_heads=1,
    ):
        super(StatementEncoder, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size

        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_tokens, embedding_size)

        self.layers = nn.ModuleList(
            [SelfAttention(
                    embedding_size,
                    num_heads,
                    ) for _ in range(num_layers)]
        )

    def forward(self, input_ids, attention_masks):
        device = input_ids.device if input_ids is not None else 'cpu'
        N, sequence_length = input_ids.shape
        positions = torch.arange(0, sequence_length)\
                         .expand(N, sequence_length)\
                         .to(device)
        input_embedding = self.word_embedding(input_ids) + \
                          self.position_embedding(positions)

        # Here, the query, key, value tensors will be the same for all layers.
        for layer in self.layers:
            output = layer(input_embedding, input_embedding, input_embedding,
                               attention_masks)

        # Here, we do not want to take a mean of the embeddings corresponding
        # to the padded tokens in the statement. An element-wise multiplication
        # of the token masks will ignore the embeddings corresponding to the
        # padded tokens in the average.
        masked_output = output * attention_masks[..., None]
        sentence_rep = torch.mean(masked_output, dim=1)
        return sentence_rep


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_size,
        intermediate_size,
        num_heads,
        dropout_rate,
        forward_activation,
    ):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)

        if forward_activation == 'gelu':
            forward_activation = torch.nn.GELU
        else:
            forward_activation = torch.nn.ReLU

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, intermediate_size),
            forward_activation(),
            nn.Linear(intermediate_size, embedding_size)
        )

        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, key, query, value, mask):
        attention = self.attention(key, query, value, mask)

        # Add residual skip-connection and normalize.
        x = self.dropout(self.norm1(attention + query))
        ff_output = self.feed_forward(x)

        # Add residual skip connection and normalize.
        output = self.dropout(self.norm2(ff_output + x))
        return output


class MLP(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        dropout_rate,
        forward_activation,
        max_stmts=8,
    ):
        super(MLP, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(4 * embedding_size, hidden_size)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Output layer
        self.fc3 = nn.Linear(hidden_size, 1)

        if forward_activation == 'gelu':
            self.forward_activation = torch.nn.GELU()
        else:
            self.forward_activation = torch.nn.ReLU()
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
  
    def forward(self, x):
        # Add first hidden layer
        x = self.forward_activation(self.fc1(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add second hidden layer
        x = self.forward_activation(self.fc2(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add output layer
        x = self.fc3(x)

        outputs = torch.sigmoid(x)
        return outputs


class ProgramDependenceModel(nn.Module):
    def __init__(
        self,
        args,
    ):
        super(ProgramDependenceModel, self).__init__()
        self.max_stmts = args.max_stmts
        self.use_stmt_types = args.use_stmt_types
        self.no_ssan = args.no_ssan
        self.no_pe = args.no_pe
        self.no_tr = args.no_tr
        self.device = args.device

        if not self.no_pe:
            self.position_embedding = nn.Embedding(self.max_stmts,
                                                   args.embedding_size)
        if self.use_stmt_types:
            self.type_embedding = nn.Embedding(args.num_stmt_types,
                                               args.embedding_size)

        if self.no_ssan:
            self.w2v_model = load_w2v_model(args.lang, args.max_stmts)
            vocab = self.w2v_model.wv.key_to_index
            self.vocab = dict(zip(vocab.values(), vocab.keys()))

        else:
            self.statement_encoder = StatementEncoder(
                    args.embedding_size,
                    args.num_layers_stmt,
                    args.vocab_size,
                    args.max_tokens,
                )

        if not self.no_tr:
            self.layers = nn.ModuleList(
                [TransformerLayer(
                        args.embedding_size,
                        args.intermediate_size,
                        args.num_heads,
                        args.dropout_rate,
                        args.forward_activation
                        ) for _ in range(args.num_layers)]
            )
        self.dropout = nn.Dropout(args.dropout_rate)

        self.cfg_mlp = MLP(args.embedding_size, args.hidden_size,
                           args.dropout_rate, args.forward_activation,
                           self.max_stmts)

        self.pdg_mlp = MLP(args.embedding_size, args.hidden_size,
                           args.dropout_rate, args.forward_activation,
                           self.max_stmts)
        self.loss_criterion = nn.BCELoss(reduction='mean')

    def _compare(self, stmts, i, j):
        return torch.cat(
            (stmts[i,],
             stmts[j,],
             stmts[i,] * stmts[j,],
             torch.abs(stmts[i,] - stmts[j,])
            ),
            dim=-1,
        )

    def forward(self, input_ids, input_masks, stmt_types, stmt_masks, labels=None):
        device = input_ids.device if input_ids is not None else 'cpu'
        input_embeddings = []

        for snippet_id, snippet_input_ids in enumerate(input_ids):
            positions = torch.arange(0, self.max_stmts).to(device)
            snippet_input_masks = input_masks[snippet_id]

            if self.no_ssan:
                input_embedding = []
                for stmt_ids in snippet_input_ids:
                    stmt_embedding = []  
                    for id in stmt_ids:
                        id = id.item()
                        if id in self.vocab and id != 0:
                            _emb = torch.tensor(self.w2v_model.wv[self.vocab[id]]).to(self.device)
                        else:
                            _emb = torch.zeros(512).to(self.device)
                        stmt_embedding.append(_emb)

                    try:
                        stmt_embedding = torch.mean(torch.stack(stmt_embedding), dim=0)
                    except TypeError:
                        # Case where there is only one token in a statement.
                        stmt_embedding = stmt_embedding[0]
                    input_embedding.append(stmt_embedding)
                input_embedding = torch.stack(input_embedding)

            else:
                input_embedding = self.statement_encoder(snippet_input_ids,
                                                         snippet_input_masks)

            if self.use_stmt_types:
                type_embedding = self.type_embedding(stmt_types[snippet_id])
                input_embedding = input_embedding + type_embedding

            if not self.no_pe:
                input_embedding = input_embedding + self.position_embedding(positions)

            input_embedding = self.dropout(input_embedding)
            input_embeddings.append(input_embedding)

        input_embeddings = torch.stack(input_embeddings)

        if not self.no_tr:
            # Here, the query, key, value tensors will be the same for all layers.
            for layer in self.layers:
                tr_outputs = layer(input_embeddings, input_embeddings,
                                   input_embeddings, stmt_masks)
        else:
            tr_outputs = input_embeddings

        if labels is not None:
            label_pairs = {
                'cfg': [],
                'pdg': [],
            }

            batch_loss = torch.tensor(0, dtype=torch.float, device=device)

            for snippet_id, snippet_stmts in enumerate(tr_outputs):
                preds_cfg, preds_pdg = [], []

                masks = stmt_masks[snippet_id]
                num_stmts = (masks == 1).sum().item()

                for i in range(num_stmts):
                    preds_cfg_ij, preds_pdg_ij = [], []
                    for j in range(num_stmts):
                        c = self._compare(snippet_stmts, i, j)
                        preds_cfg_ij.append(self.cfg_mlp(c))
                        preds_pdg_ij.append(self.pdg_mlp(c))
                    preds_cfg.append(torch.cat(preds_cfg_ij))
                    preds_pdg.append(torch.cat(preds_pdg_ij))

                preds_cfg = torch.stack(preds_cfg)
                preds_pdg = torch.stack(preds_pdg)

                labels_cfg = labels['cfg'][snippet_id][:num_stmts, :num_stmts]
                labels_pdg = labels['pdg'][snippet_id][:num_stmts, :num_stmts]

                snippet_cfg_loss = self.loss_criterion(preds_cfg, labels_cfg)
                snippet_pdg_loss = self.loss_criterion(preds_pdg, labels_pdg)
                batch_loss += snippet_cfg_loss + snippet_pdg_loss

                label_pairs['cfg'] += list(zip(
                        torch.flatten(labels_cfg).detach().cpu().numpy(),
                        torch.flatten(preds_cfg > 0.5).detach().cpu().numpy(),
                    )
                )

                label_pairs['pdg'] += list(zip(
                        torch.flatten(labels_pdg).detach().cpu().numpy(),
                        torch.flatten(preds_pdg > 0.5).detach().cpu().numpy(),
                    )
                )

            return batch_loss, label_pairs
        else:
            preds = {
                'cfg': [],
                'pdg': [],
            }
            for snippet_id, snippet_stmts in enumerate(tr_outputs):
                preds_cfg, preds_pdg = [], []

                masks = stmt_masks[snippet_id]
                num_stmts = (masks == 1).sum().item()

                for i in range(num_stmts):
                    preds_cfg_ij, preds_pdg_ij = [], []
                    for j in range(num_stmts):
                        c = self._compare(snippet_stmts, i, j)
                        preds_cfg_ij.append(self.cfg_mlp(c))
                        preds_pdg_ij.append(self.pdg_mlp(c))
                    preds_cfg.append(torch.cat(preds_cfg_ij))
                    preds_pdg.append(torch.cat(preds_pdg_ij))

                preds['cfg'].append(torch.stack(preds_cfg))
                preds['pdg'].append(torch.stack(preds_pdg))

            return preds
