import unittest
import os, sys

from template_lib import utils


class TestingExamples(unittest.TestCase):

  def test_convert_cityscapes_to_coco(self, *tmp):
    """
    Usage:

        export CUDA_VISIBLE_DEVICES=0,1
        export PORT=6006
        export TIME_STR=1
        export PYTHONPATH=./submodule
        python detectron2_exp/tests/run_detectron2.py \
          --config ./detectron2_exp/configs/detectron2.yaml \
          --command train_scratch_mask_rcnn_dense_R_50_FPN_3x_gn_2gpu \
          --outdir results/Detectron2/train_scratch_mask_rcnn_dense_R_50_FPN_3x_gn_2gpu

    :return:
    """
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
      os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if 'PORT' not in os.environ:
      os.environ['PORT'] = '6006'
    if 'TIME_STR' not in os.environ:
      os.environ['TIME_STR'] = '0' if utils.is_debugging() else '1'
    # func name
    assert sys._getframe().f_code.co_name.startswith('test_')
    command = sys._getframe().f_code.co_name[5:]
    class_name = self.__class__.__name__[7:] \
      if self.__class__.__name__.startswith('Testing') \
      else self.__class__.__name__
    outdir = f'results/{class_name}/{command}'

    argv_str = f"""
                --config ./test_run.yaml 
                --command {command}
                --outdir {outdir}
                """
    from tools import convert_cityscapes_to_coco
    convert_cityscapes_to_coco.run(argv_str)
    # run_detectron2.run(argv_str, test=True)