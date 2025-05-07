<h2>Tensorflow-Image-Segmentation-CDD-CESM-Mammogram (Updated: 2025/05/08)</h2>

This is the second experiment of Image Segmentation for CDD-CESM-Mammogram
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/19SExQHl6kbCkWm80I1ANy2v1bLRvtqF_/view?usp=sharing">
Mammogram-ImageMask-Dataset-V2.zip</a>, which was derived by us from  
<a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8">
Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)
</a>
<br>
<br>
Please see also our first experiment 
<a href="https://github.com/sarah-antillia/Image-Segmentation-CDD-CESM-Mammogram">Image-Segmentation-CDD-CESM-Mammogram
</a>
<br>
<br>
On our dataset, please refer to <a href="https://github.com/sarah-antillia/Augmentation-CDD-CESM-Mammogram-Segmentation-Dataset">
Augmentation-CDD-CESM-Mammogram-Segmentation-Dataset
</a>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P212_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P212_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P212_L_CM_MLO.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P273_L_DM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P273_L_DM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P273_L_DM_MLO.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P276_L_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P276_L_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P276_L_DM_CC.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Mammogram Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The image dataset used here has been taken from the following web site.<br>
<a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8">
Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)</a>
<br>
<br>
<b>Citations & Data Usage Policy</b><br>
Users must abide by the TCIA Data Usage Policy and Restrictions. Attribution should include references<br> 
to the following citations:<br>
<br>
<b>Data Citation</b><br>
Khaled R., Helal M., Alfarghaly O., Mokhtar O., Elkorany A., El Kassas H., Fahmy A. Categorized Digital Database<br>
for Low energy and Subtracted Contrast Enhanced Spectral Mammography images [Dataset]. (2021) The Cancer Imaging<br>
Archive. DOI:  10.7937/29kw-ae92<br>
<br>

<b>Publication Citation</b>
Khaled, R., Helal, M., Alfarghaly, O., Mokhtar, O., Elkorany, A., El Kassas, H., & Fahmy, A. Categorized contrast<br>
enhanced mammography dataset for diagnostic and artificial intelligence research. (2022) Scientific Data,<br>
Volume 9, Issue 1. DOI: 10.1038/s41597-022-01238-0<br>
<br>

<b>TCIA Citation</b><br>
Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L,<br>
Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository,<br>
Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7<br>
<br>
<h3>
<a id="2">
2 Mammogram ImageMask Dataset
</a>
</h3>
 If you would like to train this MammogramSegmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/19SExQHl6kbCkWm80I1ANy2v1bLRvtqF_/view?usp=sharing">
Mammogram-ImageMask-Dataset-V2.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Mammogram
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>Mammogram Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/Mammogram_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained MammogramTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Mammogram and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for an image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at start (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at end (60,61,62)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was stopped at epoch 62 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/train_console_output_at_epoch_62.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Mammogram</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Mammogram.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/evaluate_console_output_at_epoch_62.png" width="720" height="auto">
<br><br>Image-Segmentation-CDD-CESM-Mammogram

<a href="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Mammogram/test was not so low, and dice_coef not so high as shown below.
<br>
<pre>
loss,0.2716
dice_coef,0.6309
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Mammogram</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Mammogram.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P212_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P212_L_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P212_L_CM_MLO.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P198_L_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P198_L_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P198_L_DM_CC.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P212_L_CM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P212_L_CM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P212_L_CM_CC.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P273_L_DM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P273_L_DM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P273_L_DM_MLO.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P286_R_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P286_R_DM_CC.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P286_R_DM_CC.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/images/flipped_P306_R_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test/masks/flipped_P306_R_CM_MLO.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Mammogram/mini_test_output/flipped_P306_R_CM_MLO.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Categorized Digital Database for Low energy and Subtracted Contrast Enhanced Spectral Mammography images (CDD-CESM)
</b><br>
<a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8">
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8
</a>
<br>
<br>
<b>2. Categorized contrast enhanced mammography dataset for diagnostic and artificial intelligence research
</b><br>
Rana Khaled, Maha Helal, Omar Alfarghaly, Omnia Mokhtar, Abeer Elkorany,<br>
Hebatalla El Kassas & Aly Fahmy<br>
<a href="https://www.nature.com/articles/s41597-022-01238-0">
https://www.nature.com/articles/s41597-022-01238-0
</a>
<br>
<br>
<b>3. CDD-CESM-Dataset</b><br>
<a href="https://github.com/omar-mohamed/CDD-CESM-Dataset">
https://github.com/omar-mohamed/CDD-CESM-Dataset
</a>
<br>
<br>
<b>4. Breast Cancer Segmentation Methods: Current Status and Future Potentials</b><br>
Epimack Michael, He Ma, Hong Li, Frank Kulwa, and Jing<br>
<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321730/">
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321730/
</a>
<br>

<br>
<b>5. Image-Segmentation-CDD-CESM-Mammogram</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Image-Segmentation-CDD-CESM-Mammogram">
https://github.com/sarah-antillia/Image-Segmentation-CDD-CESM-Mammogram
</a>
<br>

