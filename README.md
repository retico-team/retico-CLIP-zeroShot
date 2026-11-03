# retico-CLIP-zeroShot
ReTiCo modules for zero-shot classification of scenes and objects using CLIP models. You give the modules a list of class labels and they classify every webcam frame (or every detected object) against those labels, without any retraining. Predictions are smoothed over time so that only stable results are committed, which makes the modules suited to dynamic, real-time environments.

If needed, you can also give the modules a small set of training images to fine-tune the predictions with the [CoOp](https://arxiv.org/abs/2109.01134) (Context Optimization) method.

## Installation and Requirements

To use the CLIP zero-shot modules you first need to install the retico-clip-zeroshot package:
* Install retico-clip-zeroshot:
```pip install retico-clip-zeroshot```

  or, for the latest development version from GitHub:
```pip install git+https://github.com/retico-team/retico-clip-zeroshot.git```

This also installs its Python dependencies, including `retico-core`, `retico-vision`, `torch` and `transformers`. The CLIP model weights are downloaded from Hugging Face automatically the first time a module runs.

**Note:** the package is installed as `retico-clip-zeroshot`, but imported as `retico_CLIP_zeroShot`.

### Hardware
* A CUDA GPU is recommended. The modules fall back to CPU automatically if no GPU is available.

## Modules

### `CLIPImageClassificationModule`
Subscribes to `ImageIU` (e.g. from `WebcamModule` in `retico_vision`) and classifies the whole scene against the given class labels, producing a `CLIPClassificationIU`.

Predictions are smoothed with an Exponential Moving Average (EMA), and a class is only committed once it has been stable for `stability_frames` frames:
* **ADD**: a new class is predicted with a confidence of at least `confidence_threshold`
* **COMMIT**: the same class has been predicted for `stability_frames` consecutive frames with sufficient confidence
* **REVOKE**: the confidence drops below `confidence_threshold`, or the predicted class changes

Example sequence:
```
Frame 1: "dog" (conf: 0.8) → ADD "dog"
Frame 2: "dog" (conf: 0.82) → (no update, building stability)
Frame 3: "dog" (conf: 0.85) → COMMIT "dog" (stable for 3 frames)
Frame 4: "cat" (conf: 0.7) → REVOKE "dog", then ADD "cat"
Frame 5: "unknown" (conf: 0.05) → REVOKE "cat"
```

#### Arguments:
* `class_labels` (list[str]): the class labels to classify against, e.g. `["dog", "cat", "car"]`; required
* `template` (str): template for the class labels, with a `{class_name}` placeholder; defaults to `"A photo of a {class_name}"`
* `model_name` (str): Hugging Face CLIP model name; defaults to `"openai/clip-vit-base-patch32"`
* `device` (str): device to run the model on; defaults to `None` (auto-detect)
* `ema_alpha` (float): EMA smoothing factor (0.0-1.0); higher is more responsive, lower is more stable; defaults to `0.3`
* `confidence_threshold` (float): minimum confidence for a class to be added; higher is more conservative, with fewer false positives; defaults to `0.1`
* `stability_frames` (int): number of consecutive frames a class must be stable before it is committed; more frames give a more stable output but a slower response; defaults to `3`
* `use_coop` (bool): use CoOp to train the token embeddings for the class labels; defaults to `False`
* `coop_examples_folder` (str): path to a folder with training images for CoOp, with one subfolder per class (e.g. `folder/dog/img1.jpg`); defaults to `None`
* `coop_n_ctx` (int): number of context tokens for CoOp; defaults to `16`
* `coop_ctx_init` (str): initial context text for CoOp (empty for random init); defaults to `""`
* `coop_epochs` (int): number of training epochs for CoOp; defaults to `50`
* `coop_lr` (float): learning rate for CoOp training; defaults to `0.002`
* `debug_prints` (bool): print debug information; defaults to `False`

### `CLIPObjectClassificationModule`
Subscribes to `ExtractedObjectsIU` (produced by `ExtractObjectsModule` in `retico_vision`) and classifies each detected object against the given class labels, producing a `CLIPObjectFeaturesIU`.

Objects are tracked across frames by the cosine similarity of their CLIP features, so each object keeps a consistent ID and has its own EMA smoothing and stability tracking. Matching is greedy and follows the order of the objects in the `ExtractedObjectsIU` input. Feature vectors are kept for up to `max_object_memory` objects, and the state of objects that disappear is removed automatically.
* **ADD**: the first objects are detected, a new object appears, or an existing object changes class
* **COMMIT**: every detected object has kept the same class for `stability_frames` consecutive frames
* **REVOKE**: all objects disappear from the scene, or the classifications become unreliable (low confidence)

Example sequence:
```
Frame 1: [obj1: "dog" (conf: 0.8)] → ADD objects
Frame 2: [obj1: "dog" (conf: 0.82)] → (building stability)
Frame 3: [obj1: "dog" (conf: 0.85), obj2: "car" (conf: 0.9)] → ADD objects (new car detected)
Frame 4: [obj1: "dog" (conf: 0.87), obj2: "car" (conf: 0.91)] → (building stability)
Frame 5: [obj1: "dog" (conf: 0.86), obj2: "car" (conf: 0.89)] → COMMIT objects (all stable)
Frame 6: [] → REVOKE objects (all disappeared)
```

#### Arguments:
Takes all the arguments of `CLIPImageClassificationModule` above, plus:

**Object tracking**
* `similarity_threshold` (float): cosine similarity threshold for matching objects across frames; higher means stricter matching and more new IDs; defaults to `0.8`
* `max_object_memory` (int): the maximum number of objects to remember across frames; defaults to `10`

### `ClassificationConsumer` and `ObjectClassificationConsumer`
Debug modules (in `retico_CLIP_zeroShot.debug_modules`) that print the output of `CLIPImageClassificationModule` and `CLIPObjectClassificationModule`. They take no arguments.

### Example
**Note:** The example runner uses `WebcamModule` from `retico_vision` to classify the whole scene.

```python
from retico_core import *
from retico_vision.vision import WebcamModule

from retico_CLIP_zeroShot.image_classification import CLIPImageClassificationModule
from retico_CLIP_zeroShot.debug_modules import ClassificationConsumer

video = WebcamModule()

template = "A scene from {class_name}."
classes = ["outdoor", "indoor", "kitchen", "bedroom", "living room",
           "office", "bathroom", "restaurant", "street", "park",
           "beach", "forest", "urban area", "rural area", "cityscape"]
classifier = CLIPImageClassificationModule(
    template=template,
    class_labels=classes,
    model_name="openai/clip-vit-base-patch32",
    ema_alpha=0.3,
    confidence_threshold=0.15,
    stability_frames=3,
)
debug_classifier = ClassificationConsumer()

video.subscribe(classifier)
classifier.subscribe(debug_classifier)

network.run(video)

input("Network is running. Press Enter to stop...")

network.stop(video)
```

### Example 2
**Note:** The example runner uses `WebcamModule` and `ExtractObjectsModule` from `retico_vision` and `Yolov11` from [retico-yolov11](https://github.com/retico-team/retico-yolov11) (`pip install retico-yolov11`) to detect the objects. `ExtractObjectsModule` lets you choose how many objects are extracted from each frame.

```python
from retico_core import *
from retico_vision.vision import WebcamModule, ExtractObjectsModule

from retico_yolov11.yolov11 import Yolov11

from retico_CLIP_zeroShot.object_classification import CLIPObjectClassificationModule
from retico_CLIP_zeroShot.debug_modules import ObjectClassificationConsumer

video = WebcamModule()
yolo = Yolov11()
extractor = ExtractObjectsModule(num_obj_to_display=5)

object_classifier = CLIPObjectClassificationModule(
    model_name="openai/clip-vit-base-patch32",
    template="A photo of a {class_name}.",
    class_labels=["cat", "dog", "car", "bicycle", "person", "mouse", "keyboard", "phone", "laptop", "book"],
    ema_alpha=0.4,
    confidence_threshold=0.2,
    stability_frames=2,
    debug_prints=False,
)
debug_obj_classifier = ObjectClassificationConsumer()

video.subscribe(yolo)
yolo.subscribe(extractor)
extractor.subscribe(object_classifier)
object_classifier.subscribe(debug_obj_classifier)

network.run(video)

input("Network is running. Press Enter to stop...")

network.stop(video)
```

Citation
```
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}

@article{zhou2022coop,
  title={Learning to Prompt for Vision-Language Models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  journal={International Journal of Computer Vision (IJCV)},
  year={2022}
}
```