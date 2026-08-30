# Baseline weight storage

The organizer baseline weights are intentionally excluded from GitHub because
they exceed GitHub's normal Git/LFS object limits. The reviewed local copies
are mirrored in the private Hugging Face model repository below.

- repository: <https://huggingface.co/datahj/deepvoice2608>
- immutable revision: `8671000ad992b8e9e58558c6869126fb7b635a88`
- storage tree: <https://huggingface.co/datahj/deepvoice2608/tree/8671000ad992b8e9e58558c6869126fb7b635a88/storage>

| Local path under `model/` | Hugging Face object | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `df_arena_1b/pytorch_model.bin` | [`storage/df_arena_1b/pytorch_model.bin`](https://huggingface.co/datahj/deepvoice2608/resolve/8671000ad992b8e9e58558c6869126fb7b635a88/storage/df_arena_1b/pytorch_model.bin) | 4,591,794,734 | `780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988` |
| `htdemucs/955717e8-8726e21a.th` | [`storage/htdemucs/955717e8-8726e21a.th`](https://huggingface.co/datahj/deepvoice2608/resolve/8671000ad992b8e9e58558c6869126fb7b635a88/storage/htdemucs/955717e8-8726e21a.th) | 84,141,911 | `8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4` |
| `panns/Cnn14_mAP=0.431.pth` | [`storage/panns/Cnn14_mAP=0.431.pth`](https://huggingface.co/datahj/deepvoice2608/resolve/8671000ad992b8e9e58558c6869126fb7b635a88/storage/panns/Cnn14_mAP=0.431.pth) | 327,428,481 | `0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31` |

The repository is private. Authenticate with `hf auth login`, then restore and
verify all three files without executing them:

```powershell
powershell -ExecutionPolicy Bypass -File .\format\baseline_submit\model\restore_from_hf.ps1
```

The restore script downloads to an isolated temporary directory, validates
the exact byte size and SHA-256 from `SHA256SUMS.txt`, and only then copies a
missing file into the baseline model directory. It refuses to overwrite a
different local file.
