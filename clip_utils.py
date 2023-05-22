import torch
import numpy as np
import os
import random
import sys

sys.path.append('CLIP2Video')
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.modeling import CLIP2Video
# from evaluation.eval import eval_epoch

# checkpoint = "https://drive.google.com/drive/folders/1LKMUZFf9EAxFbGShlA22eUCeGKC8DWx4?usp=sharing"
model_num = "2"
checkpoint ="/content/vm-dataset-collector/"
cross_model = "cross-base"
from dataclasses import dataclass

@dataclass
class TaskConfig:
  random_thing: str = ""
  clip_path: str = "/content/vm-dataset-collector/clip.pt"
  temporal_type: str = "TDB"
  sim_type: str = "seqTransf"
  center_type: str = "TAB"
  temporal_proj: str = "sigmoid_selfA"


  vocab_size: int = 49408

  val_csv: str = "data/.val.csv"
  data_path: str = "data/caption.pickle"
  features_path: str = "data/videos_feature.pickle"
  num_thread_reader: int = 1
  batch_size_val: int = 3500
  seed: int = 42
  max_words: int = 32
  max_frames: int = 100
  feature_framerate: int = 1
  output_dir: str = None
  cross_model: str = "cross-base"
  do_lower_case: bool = False
  n_gpu: int = 1
  cache_dir: str = ""
  fp16: bool = False
  fp16_opt_level: str = "O1"
  cross_num_hidden_layers: int = 4
  # sim_type: str = "meanP"
  checkpoint: str = ""
  model_num: str = ""
  local_rank: int = 0
  datatype: str = "msrvtt"
  vocab_size: int = 49408
  # temporal_type: str = ""
  # temporal_proj: str = ""
  # center_type: str = ""
  centerK: int = 5
  center_weight: float = 0.5
  center_proj: str = ""

  num_thread_reader: int  =2
  # data_path: str =  "${"DATA_PATH} \
  # features_path: str =  "["frame path] \
  output_dir: str =  "./"
  max_words: str =  "32" 
  max_frames: str =  "12" 
  batch_size_val: str =  "64" 
  datatype: str =  "msvd" 
  feature_framerate: int =  1 
  # sim_type: str =  "seqTransf" 
  # checkpoint: str =  "${"CHEC
  do_eval: bool = True
  # model_num: str =  "${"MODEL_NUM} 
  temporal_type: str =  "TDB" 
  temporal_proj: str =  "sigmoid_selfA" 
  center_type: str =  "TAB" 
  centerK: int =  5 
  center_weight: float =  0.5
  center_proj: str =  "TAB_TDB"

dummy_task_config = TaskConfig()

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_model():
    """Initialize model.

    if location of args.init_model exists, model will be initialized from the pretrained model.
    if no model exists, the training will be initialized from CLIP's parameters.

    Args:
        args: the hyper-parameters
        devices: cuda

    Returns:
        model: the initialized model

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # resume model if pre-trained model exist.
    model_file = os.path.join(checkpoint, "pytorch_model.bin.{}".format(model_num))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        # if args.local_rank == 0:
        #     logger.info("Model loaded from %s", model_file)
    else:
        model_state_dict = None
        # if args.local_rank == 0:
        #     logger.info("Model loaded fail %s", model_file)

    # Prepare model
    model = CLIP2Video.from_pretrained(cross_model, cache_dir=None, state_dict=model_state_dict,
                                       task_config=dummy_task_config)
    model.to(device)

    return model

clip_model = init_model()

clip_model.eval()

from dataloaders.rawframe_util import RawFrameExtractor

frame_extractor = RawFrameExtractor(framerate=20)


def load_frames(video_frames_dir, framerate=10, max_frame=1200):
  """
  This will return (max_frame)//(framerate frames)

  If the input sequence of the transformer can go to 512, then the max_frame can go to 5120 with a framerate of 10

  That means we can process videos up to 3:55 in length, with 24fps
  """
  # print("Extracting ", video_frames_dir)
  video_data = frame_extractor.get_video_data(video_frames_dir, max_frame=1200)

  return video_data

import os
from concurrent import futures

def extract_frames(video_name, out_folder, fps=5):
    if os.path.exists(out_folder):
        return
        # os.system('rm -rf ' + out_folder + '/*')
        # os.system('rm -rf ' + out_folder)
    os.makedirs(out_folder)
    cmd = 'ffmpeg -v 0 -i %s -r %d -q 0 %s/%s.jpg' % (video_name, fps, out_folder, '%08d')
    os.system(cmd)

def process(line):
    # print(line)
    mp4_name, folder_frame = line
    extract_frames(mp4_name, folder_frame, fps=20)


# if __name__ == "__main__":


def extract_frames_from_videos(input_path_video, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)


    # TODO: Use glob here
    # mp4_file = list(filter(lambda x: x[-3:] == "mp4", os.listdir(input_path)))

    # mp4_file = mp4_file[:5]

    mp4_file = [input_path_video]

    input_path_dir = os.path.dirname(input_path_video)

    # print("Files to be extracted", mp4_file)
    lines = [(os.path.join(input_path_dir, mp4), os.path.join(output_path, mp4.split(".")[0])) for mp4 in mp4_file]

    output_directories = [x[1] for x in lines]

    # multi thread
    with futures.ProcessPoolExecutor(max_workers=5) as executer:
        fs = [executer.submit(process, line) for line in lines]
    print("done")

    return output_directories

import gc

clip_model.frame2t_attention.cuda()
def _clip_embedding(video_data):
  # torch.set_default_tensor_type('torch.cuda.FloatTensor')


  if video_data.shape[0] < 60:
    try:
        video_data = torch.cat([video_data, torch.zeros((60 - video_data.shape[0], 3, 224, 224))])
    except:
        print("ERROR with loaded frames dimensions, skipping this video")
        return None

  max_frames = video_data.shape[0]
  video_mask = torch.tensor(np.zeros((1, max_frames), dtype=np.long)).to(device)

#   if max_frames < 60:
     


  visual_output = clip_model.get_visual_output(video_data.to(device), video_mask.to(device), shaped=True)
  # print(visual_output.shape)

   # with torch.device('cuda'):
  visual_output = visual_output.contiguous()

  # obtain the basic embedding
  visual_output_original = visual_output # batch_size * 12 * 512

  # difference-enhanced token obtained by TDB
  # print(visual_output.shape)
  # assert False
  visual_output, frame_position_embeddings, type_embedding, temporal_video_mask = clip_model.temporal_difference_block(
      visual_output, video_mask)
  
  # print(visual_output.shape)
  # print(visual_output)

  # obtain the output of transformer
  visual_output = visual_output + frame_position_embeddings + type_embedding # batch_size * 12 * 512
  extended_video_mask = (1.0 - temporal_video_mask.unsqueeze(1)) * -1000000.0 # batch_size * 1* 12
  extended_video_mask = extended_video_mask.expand(-1, temporal_video_mask.size(1), -1) # batch_size * 12 * 12
  visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND # 12 * batch_size * 512
  visual_output = clip_model.transformerClip(visual_output, extended_video_mask) #12 * batch_size * 512
  visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD # batch_size * 12 * 512

  # print(visual_output)

  # select the output of frame token for final video representation
  frame_position_id = torch.arange(start=0, end=visual_output.size()[1], step=2, dtype=torch.long,
                                    device=visual_output.device)
  visual_output = visual_output[:, frame_position_id, :]
  visual_output = visual_output + visual_output_original
  visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

  # torch.set_default_tensor_type('torch.FloatTensor')


  return visual_output

def clip_embedding(x):
  # Do the embeddings one by one:
  # print(x.shape)
  # result = []
  # for i in range(x.shape[0]):
  #   current = x[i]
  #   print(current.shape)
  #   result.append(_clip_embedding(current))
  #   torch.cuda.empty_cache()
  # result = torch.cat([i.unsqueeze(0) for i in result])
  result = _clip_embedding(x)
  gc.collect()
  return result


def video_to_clip_embedding(video_path):
    extracted_dir = os.path.dirname(video_path)
   
    extracted_frames_dir = extract_frames_from_videos(video_path, extracted_dir)

    video_data = load_frames(extracted_frames_dir)

    embedding = clip_embedding(video_data["video"])

    if embedding is None:
       return None

    orig_no_ext = ".".join(video_path.split(".")[:-1])

    target_path = orig_no_ext + ".tensor"
    torch.save(embedding, target_path)

    return target_path
