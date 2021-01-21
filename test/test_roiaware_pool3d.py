import torch
import pytest

from lib.utils.voxelnet_aug import check_inside_points


def test_points_in_boxes():
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    boxes = torch.tensor(
        # x, y, z, l, h, w
        [[1.0, 3.0, 2.0, 5.0, 6.0, 4.0, 0.3],
         [-10.0, 23.0, 16.0, 10, 20, 20, 0.5]
         ],
        dtype=torch.float32
    ).numpy()  # boxes (m, 7) with bottom center in lidar coordinate

    pts = torch.tensor(
        [[1, 2, 3.3], [1.2, 2.5, 3.0], [0.8, 2.1, 3.5], [1.6, 2.6, 3.6],
         [0.8, 1.2, 3.9], [-9.2, 21.0, 18.2], [3.8, 7.9, 6.3],
         [4.7, 3.5, -12.2], [3.8, 7.6, -2], [-10.6, -12.9, -20], [-16, -18, 9],
         [-21.3, -52, -5], [0, 0, 0], [6, 7, 8], [-2, -3, -4]],
        dtype=torch.float32).numpy()  # points (n, 3) in lidar coordinate

    expected_point_indices = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.bool).numpy().T

    point_indices = check_inside_points(points=pts, cur_boxes=boxes)
    assert point_indices.shape == torch.Size([15, 2])
    assert (point_indices == expected_point_indices).all()


if __name__ == '__main__':
    test_points_in_boxes()