import os
import json
import torch
import numpy as np
from omegaconf import DictConfig

from anim import bvh
from anim import quat
from anim.txform import *
from audio.audio_files import read_wavfile
from data_pipeline import preprocess_animation
from data_pipeline import preprocess_audio
from helpers import split_by_ratio
from utils import write_bvh


class ZeroEggsInference(object):
    """Generate stylized gesture from raw audio and style example (ZEGGS)
    """
    def __init__(self, network_directory_path, config_directory_path):
        """
        Args:
            network_directory_path: Directory Path to the networks
        """
        # Define device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load networks
        networks = self._load_networks(network_directory_path)
        self.speech_encoder_network = networks[0]
        self.decoder_network = networks[1]
        self.style_encoder_network = networks[2]

        # Load configs
        self.stat_data, self.data_pipeline_conf, self.details = self._load_data_configs(config_directory_path)

    def generate_gesture_from_audio_file(self, audio_file_path, style_bvh_path, style_encoding_type="example", style_name=None):
        """
        Args:
            audio_file_path: Path to audio file.
            style_bvh_path: Path to style bvh file.
            style_encoding_type: How to encode the style. Either "example" or "label". Defaults to "example". If "label" type, specify the style_name argument.
            style_name: Style name. Style categories covers 19 different motion styles and can be selected from the following.
                "Agreement", "Angry", "Disagreement", "Distracted", "Flirty", "Happy", "Laughing",
                "Oration", "Neutral", "Old", "Pensive", "Relaxed", "Sad", "Sarcastic", "Scared",
                "Sneaky", "Threatening", "Tired"
        Returns:
            bvh_output: Body Gestures Output in BVH format
        """
        with torch.no_grad():
            _, audio_data = read_wavfile(
                audio_file_path, 
                rescale=True, desired_fs=16000, out_type="float32"
            )

            n_frames = int(round(60.0 * (len(audio_data) / 16000)))
            audio_features = torch.as_tensor(
                    preprocess_audio(
                        audio_data,
                        60,
                        n_frames,
                        self.data_pipeline_conf["audio_conf"],
                        feature_type=self.data_pipeline_conf["audio_feature_type"],
                    ),
                    device=self.device,
                    dtype=torch.float32,
            ).detach().cpu().numpy()

            # Normalize Audio Input
            audio_features = (audio_features[np.newaxis] - self.stat_data["audio_input_mean"]) / self.stat_data["audio_input_std"]
            audio_features = torch.as_tensor(audio_features, device=self.device, dtype=torch.float32)
            
            # 1) Speech Encoding
            speech_encoding = self.speech_encoder_network(audio_features)

            # 2) Style Encoding
            style_encoding, animation_data = self._create_style_encoding(
                style_bvh_path=style_bvh_path,
                style_encoding_type=style_encoding_type,
                style_name=style_name,
                temperature=1.0
            )

            # 3) Decode and Generate Gestures
            bvh_output = self._make_bvh_outputs(
                speech_encoding=speech_encoding,
                style_encoding=style_encoding,
                animation_data=animation_data
            )
            return bvh_output

    def generate_gesture_from_audio_chunk(self, audio_chunk, style_bvh_path, style_encoding_type="example", style_name=None):
        """
        Args:
            audio_chunk: Numpy array of audio chunk data
            style_bvh_path: Path to style bvh file.
            style_encoding_type: How to encode the style. Either "example" or "label". Defaults to "example". If "label" type, specify the style_name argument.
            style_name: Style name. Style categories covers 19 different motion styles and can be selected from the following.
                "Agreement", "Angry", "Disagreement", "Distracted", "Flirty", "Happy", "Laughing",
                "Oration", "Neutral", "Old", "Pensive", "Relaxed", "Sad", "Sarcastic", "Scared",
                "Sneaky", "Threatening", "Tired"
        Returns:
            bvh_output: Body Gestures Output in BVH format
        """
        n_frames = int(round(60.0 * (len(audio_chunk) / 16000)))

        with torch.no_grad():
            audio_features = torch.as_tensor(
                    preprocess_audio(
                        audio_chunk,
                        60,
                        n_frames,
                        self.data_pipeline_conf["audio_conf"],
                        feature_type=self.data_pipeline_conf["audio_feature_type"],
                    ),
                    device=self.device,
                    dtype=torch.float32,
            )
            # Normalize Audio Input
            audio_features = (audio_features[np.newaxis] - self.stat_data["audio_input_mean"]) / self.stat_data["audio_input_std"]
            
            # 1) Speech Encoding
            speech_encoding = self.speech_encoder_network(audio_features)

            # 2) Style Encoding
            style_encoding, animation_data = self._create_style_encoding(
                style_bvh_path=style_bvh_path,
                style_encoding_type=style_encoding_type,
                style_name=style_name,
                temperature=1.0
            )

            # 3) Decode and Generate Gestures
            bvh_output = self._make_bvh_outputs(
                speech_encoding=speech_encoding,
                style_encoding=style_encoding,
                animation_data=animation_data
            )
            return bvh_output

    def _make_bvh_outputs(self, speech_encoding, style_encoding, animation_data):
        if style_encoding.dim() == 2:
            style_encoding = style_encoding.unsqueeze(1).repeat((1, speech_encoding.shape[1], 1))

        parents = torch.as_tensor(self.details["parents"], dtype=torch.long, device=self.device)
        dt = self.details["dt"]
        bone_names = self.details["bone_names"]

        anim_input_mean = torch.as_tensor(
            self.stat_data["anim_input_mean"], dtype=torch.float32, device=self.device
        )
        anim_input_std = torch.as_tensor(
            self.stat_data["anim_input_std"], dtype=torch.float32, device=self.device
        )
        anim_output_mean = torch.as_tensor(
            self.stat_data["anim_output_mean"], dtype=torch.float32, device=self.device
        )
        anim_output_std = torch.as_tensor(
            self.stat_data["anim_output_std"], dtype=torch.float32, device=self.device
        )

        root_pos, root_rot, root_vel, root_vrt, lpos, ltxy, lvel, lvrt, gaze_pos = animation_data
        (
            V_root_pos,
            V_root_rot,
            V_root_vel,
            V_root_vrt,
            V_lpos,
            V_ltxy,
            V_lvel,
            V_lvrt,
        ) = self.decoder_network(
            root_pos[0][np.newaxis], 
            root_rot[0][np.newaxis], 
            root_vel[0][np.newaxis], 
            root_vrt[0][np.newaxis],
            lpos[0][np.newaxis], 
            ltxy[0][np.newaxis], 
            lvel[0][np.newaxis], 
            lvrt[0][np.newaxis],
            gaze_pos[0:0+1].repeat_interleave(speech_encoding.shape[1], dim=0)[np.newaxis],
            speech_encoding, 
            style_encoding, 
            parents,
            anim_input_mean,
            anim_input_std, 
            anim_output_mean, 
            anim_output_std, 
            dt
        )

        V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).detach().cpu().numpy())
        bvh_info = write_bvh(
            V_root_pos[0].detach().cpu().numpy(),
            V_root_rot[0].detach().cpu().numpy(),
            V_lpos[0].detach().cpu().numpy(),
            V_lrot[0],
            parents=parents.detach().cpu().numpy(),
            names=bone_names,
            order="zyx",
            dt=dt,
            start_position=np.array([0, 0, 0]),
            start_rotation=np.array([1, 0, 0, 0]),
        )
        return bvh_info

    def _create_style_encoding(self, style_bvh_path, style_encoding_type="example", temperature=1.0, style_name=None):
        anim_data = bvh.load(style_bvh_path)
        anim_fps = int(np.ceil(1 / anim_data["frametime"]))
        assert anim_fps == 60

        # Extracting features
        (
            root_pos,
            root_rot,
            root_vel,
            root_vrt,
            lpos,
            lrot,
            ltxy,
            lvel,
            lvrt,
            cpos,
            crot,
            ctxy,
            cvel,
            cvrt,
            gaze_pos,
            gaze_dir,
        ) = preprocess_animation(anim_data)

         # convert to tensor
        nframes = len(anim_data["rotations"])
        root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=self.device)
        root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=self.device)
        root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=self.device)
        root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=self.device)
        lpos = torch.as_tensor(lpos, dtype=torch.float32, device=self.device)
        ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=self.device)
        lvel = torch.as_tensor(lvel, dtype=torch.float32, device=self.device)
        lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=self.device)
        gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=self.device)

        if style_encoding_type == "example":
            S_root_vel = root_vel.reshape(nframes, -1)
            S_root_vrt = root_vrt.reshape(nframes, -1)
            S_lpos = lpos.reshape(nframes, -1)
            S_ltxy = ltxy.reshape(nframes, -1)
            S_lvel = lvel.reshape(nframes, -1)
            S_lvrt = lvrt.reshape(nframes, -1)
            example_feature_vec = torch.cat(
                [
                    S_root_vel,
                    S_root_vrt,
                    S_lpos,
                    S_ltxy,
                    S_lvel,
                    S_lvrt,
                    torch.zeros_like(S_root_vel),
                ],
                dim=1,
            ).detach().cpu().numpy()

            example_feature_vec = (example_feature_vec - self.stat_data["anim_input_mean"]) / self.stat_data["anim_input_std"]
            example_feature_vec = torch.as_tensor(example_feature_vec, dtype=torch.float32, device=self.device)

            style_encoding, _, _ = self.style_encoder_network(
                example_feature_vec[np.newaxis], temperature
            )
        
        elif (style_encoding_type == "label") & (style_name is not None):
            raise NotImplementedError
            # n_labels = len(self.details["label_names"])
            # style_index = self.details["label_names"].index(style_name)
            # style_encoding = torch.zeros((1, n_labels), dtype=torch.float32, device=self.device)
            # style_encoding[0, style_index] = 1.0

        animation_data = (
            root_pos, root_rot, root_vel, root_vrt,
            lpos, ltxy, lvel, lvrt, gaze_pos
        )

        return style_encoding, animation_data

    def _load_data_configs(self, config_dir):
        # Load configs
        stat_data_path = os.path.join(config_dir, "stats.npz")
        data_pipeline_path = os.path.join(config_dir, "data_pipeline_conf.json")
        data_definition_path = os.path.join(config_dir, "data_definition.json")

        stat_data = np.load(stat_data_path)
        data_pipeline_conf = DictConfig(json.load(open(data_pipeline_path, "r")))
        data_definition = json.load(open(data_definition_path, "r"))

        return stat_data, data_pipeline_conf, data_definition

    def _load_networks(self, network_dir):
        # Load networks
        encoder_weights_path = os.path.join(network_dir, "speech_encoder.pt")
        decoder_weights_path = os.path.join(network_dir, "decoder.pt")
        style_encoder_path = os.path.join(network_dir, "style_encoder.pt")

        speech_encoder_network = torch.load(
            encoder_weights_path, map_location=self.device
        )
        decoder_network = torch.load(
            decoder_weights_path, map_location=self.device
        )
        style_encoder_network = torch.load(
            style_encoder_path, map_location=self.device
        )
        speech_encoder_network.eval()
        decoder_network.eval()
        style_encoder_network.eval()

        return speech_encoder_network, decoder_network, style_encoder_network



if __name__ == "__main__":
    network_dir = "../data/outputs/v1/saved_models/"
    config_dir = "../data/processed_v1/"

    zeggs_module = ZeroEggsInference(
        network_directory_path=network_dir,
        config_directory_path=config_dir
    )

    # Style Encoding Type
    ## "Example" based
    audio_file_path = "../data/samples/067_Speech_2_x_1_0.wav"
    bvh_file_path = "../data/samples/067_Speech_2_x_1_0.bvh"

    bvh_output = zeggs_module.generate_gesture_from_audio_file(
        audio_file_path, bvh_file_path, style_encoding_type="example"
    )

    ## "Label" based
    # bvh_output = zeggs_module.generate_gesture_from_audio_file(
        # audio_file_path, bvh_file_path, style_encoding_type="label", style_name="Sad"
    # )

    # bvh.save("./test.bvh", bvh_output)
