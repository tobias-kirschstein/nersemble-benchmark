ASSETS = {
    "nvs":
        {
            "per_person":
                {
                    "calibration": "{p_id:03d}/calibration/camera_params.json"
                },
            "per_cam":
                {
                    "images": "{p_id:03d}/sequences/{seq_name:}/images/cam_{serial:}.mp4",
                    "alpha_maps": "{p_id:03d}/sequences/{seq_name:}/alpha_maps/cam_{serial:}.mp4",
                },
            "per_timestep":
                {
                    "pointclouds": "{p_id:03d}/sequences/{seq_name:}/pointclouds/frame_{timestep:05d}.pcd"
                }
        },
    "mono_flame_avatar":
        {
            "per_person":
                {
                    "calibration": "{p_id:03d}/calibration/camera_params.json"
                },
            "per_sequence":
                {
                    "flame2023_tracking": "{p_id:03d}/sequences/{seq_name:}/tracking/flame2023_tracking.npz"
                },
            "per_cam":
                {
                    "images": "{p_id:03d}/sequences/{seq_name:}/images/cam_{serial:}.mp4",
                    "alpha_maps": "{p_id:03d}/sequences/{seq_name:}/alpha_maps/cam_{serial:}.mp4",
                },
            "test_assets":
                [
                    "flame2023_tracking"
                ]
        }
}

SERIALS = ["222200042", "222200044", "222200046", "222200040", "222200036", "222200048", "220700191", "222200041",
           "222200037", "222200038", "222200047", "222200043", "222200049", "222200039", "222200045", "221501007"]

NERSEMBLE_ACCESS_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScYsXR8NVCi4nvmCbFNL0P9swsGodMnbntUJeFejtuKUMsY7Q/viewform"

# ----------------------------------------------------------
# NVS Benchmark
# ----------------------------------------------------------

BENCHMARK_NVS_IDS_AND_SEQUENCES = [
    (388, 'GLASSES'),
    (422, 'EXP-2-eyes'),
    (443, 'FREE'),
    (445, 'EXP-6-tongue-1'),
    (475, 'HAIR')
]

BENCHMARK_NVS_HOLD_OUT_SERIALS = ["222200046", "222200037", "222200039"]  # left, center, right
BENCHMARK_NVS_TRAIN_SERIALS = [serial for serial in SERIALS if serial not in BENCHMARK_NVS_HOLD_OUT_SERIALS]

# ----------------------------------------------------------
# Mono FLAME Avatar Benchmark
# ----------------------------------------------------------

BENCHMARK_MONO_FLAME_AVATAR_IDS = [393, 404, 461, 477, 486]
BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TRAIN = [
    "EXP-1-head",
    "EXP-2-eyes",
    "EXP-3-cheeks+nose",
    "EXP-4-lips",
    "EXP-5-mouth",
    "EXP-8-jaw-1",
    "EXP-9-jaw-2",

    "EMO-2-surprise+fear",
    "EMO-3-angry+sad",
    "EMO-4-disgust+happy",

    "SEN-01-cramp_small_danger",
    "SEN-02-same_phrase_thirty_times",
    "SEN-03-pluck_bright_rose",
    "SEN-04-two_plus_seven",
    "SEN-05-glow_eyes_sweet_girl",
    "SEN-06-problems_wise_chief",
    "SEN-07-fond_note_fried",
    "SEN-08-clothes_and_lodging",
]

BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST = [
    "FREE",
    "EMO-1-shout+laugh",
    "SEN-09-frown_events_bad",
    "SEN-10-port_strong_smokey"
]

BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES = BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TRAIN + BENCHMARK_MONO_FLAME_AVATAR_SEQUENCES_TEST

BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL = "222200037"
BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS = ["222200046", "220700191", "222200039"]  # left, center, right
BENCHMARK_MONO_AVATAR_SERIALS = [BENCHMARK_MONO_FLAME_AVATAR_TRAIN_SERIAL] + BENCHMARK_MONO_FLAME_AVATAR_HOLD_OUT_SERIALS
