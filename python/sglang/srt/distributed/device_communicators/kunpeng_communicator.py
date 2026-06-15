import logging
import os
from typing import Optional

import torch
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _kunpeng_tp_enabled() -> bool:
    return os.environ.get("SGLANG_USE_KUNPENG_TP", "0") == "1"


class KunpengCommunicator:
    """TP collectives backed by the custom sglang_kupl (KUTACC/KUPL) runtime.

    rank/world_size/local_rank are taken from sglang's GroupCoordinator and
    passed straight into init_runtime; the C++ side must build its KUTACC
    domain from these values rather than inferring rank from MPI_Comm_rank,
    otherwise the gloo and KUTACC rank spaces can diverge and corrupt data.
    """

    def __init__(
        self,
        group: ProcessGroup,
        rank_in_group: int,
        world_size: int,
        local_rank: int,
    ):
        if not _kunpeng_tp_enabled() or world_size == 1:
            self.disabled = True
            return

        try:
            import sglang_kupl  # noqa: F401  triggers torch.ops.sglang_kupl registration
        except ImportError as e:
            raise ImportError(
                "SGLANG_USE_KUNPENG_TP=1 but the sglang_kupl extension is not "
                "importable. Build it from Kpllminfer/sglang_kupl first."
            ) from e

        self.disabled = False
        self.group = group
        self.world_size = world_size

        comm_cores = os.environ.get("KUNPENG_COMM_CORES", "")
        # Pure-CPU inference: device_id is advisory (e.g. NUMA binding). Default
        # to local_rank; override via KUNPENG_DEVICE_ID if the runtime needs it.
        device_id = int(os.environ.get("KUNPENG_DEVICE_ID", local_rank))

        self.handle = torch.ops.sglang_kupl.init_runtime(
            rank_in_group, world_size, local_rank, device_id, comm_cores
        )
        logger.info(
            "KunpengCommunicator initialized: rank_in_group=%d world_size=%d "
            "local_rank=%d device_id=%d comm_cores=%r handle=%d",
            rank_in_group,
            world_size,
            local_rank,
            device_id,
            comm_cores,
            self.handle,
        )

    def __del__(self):
        """Destructor: clean up runtime resources when communicator is garbage collected.

        This is critical for preventing MPI/KUPL/KUTACC resource leaks when the
        server restarts or workers are re-initialized.
        """
        self.destroy()

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # tp_all_reduce is out-of-place: (handle, input, output).
        out = torch.empty_like(x)
        torch.ops.sglang_kupl.tp_all_reduce(self.handle, x, out)
        return out

    def all_gather(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += x.dim()
        input_size = x.size()

        # KUTACC tp_all_gather 总是沿最后一维 concat: (batch, input_size) -> (batch, input_size*world)
        output_size = tuple(input_size[:-1]) + (input_size[-1] * self.world_size,)
        out = torch.empty(output_size, dtype=x.dtype, device=x.device)
        torch.ops.sglang_kupl.tp_all_gather(self.handle, out, x)

        # DeepSeek R1 INT8 推理只用 dim=-1 (hidden_dim concat)。
        # 如果将来需要支持其他 dim，需要在 KUTACC gather 后手动 transpose。
        if dim != x.dim() - 1:
            raise NotImplementedError(
                f"kunpeng all_gather only supports dim=-1 (last dim), got dim={dim}. "
                "DeepSeek R1 INT8 inference does not need other dims."
            )
        return out

    def barrier(self) -> None:
        if not self.disabled:
            torch.ops.sglang_kupl.tp_barrier(self.handle)

    def destroy(self) -> None:
        """Destroy the runtime and free all MPI/KUPL/KUTACC resources.

        This method is idempotent — calling it multiple times is safe.
        It's automatically called by __del__, but can also be called explicitly
        during graceful shutdown.
        """
        if not self.disabled and hasattr(self, 'handle'):
            try:
                torch.ops.sglang_kupl.destroy_runtime(self.handle)
            except Exception as e:
                # Log but don't crash during cleanup
                logger.warning("Failed to destroy kunpeng runtime: %s", e)
            finally:
                self.disabled = True
                delattr(self, 'handle')
