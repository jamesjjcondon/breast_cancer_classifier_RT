My background is medicine, not computer science, and it shows: the code is ugly and needs a lot of optimisation, but it works...

This repo is forked and modified from Wu et al. / NYU's [nyukat/breast_cancer_classifier](https://github.com/nyukat/breast_cancer_classifier)
under the terms of the GNU AGPLv3 license.
 
This is largely a personal development repo for our paper 'Replication of deep learning for screening mammography: Reduced performance mitigated by retraining on local data*, availalble in preprint at [medRxiv](https://www.medrxiv.org/content/10.1101/2021.05.28.21257892v1)
in which we show that retraining a convolutional neural network (a type of artificial intelligence) developed at New York University, using local data, can improve breast cancer diagnosis in mammograms. 

We are not able to share the BSSA data, or models at this time. 
 
#### Breast Cancer 
Breast cancer is responsible for 1 in 4 cancer deaths in women worldwide.(5) Delays in diagnosis and treatment, compared to earlier detection, are associated with adverse clinical outcomes and treatments that can decrease quality of life.(6)(7)(8)
Some authors have reported inconsistent performance of deep learning (DL) systems for mammogram classification on independent datasets.(1)(2)

#### BSSA data
(flowchart)

#### BSSA Pipeline
1. Images (dicoms) and data transfer from BSSA to secure storage
2. Slicing patient data into train/val/test splits and downloading dicoms
3. Monochrome inversion, size-matching to NYU sample images and cropping black-space
4. Generate benign and malignant heatmaps (see below) 
5. Training Models with centre crop, crop windows and crop random noise augmentation matched to NYU pipeline
6. Inference / model predictions
7. Evaluation comparing:
   - original "static" NYU models versus trained from scratch versus trained with NYU-weights / transfer learning
   - for both image-only (NYU1) and image with heatmaps (NYU2)

#### Methods and Materials
All clients with biopsy and surgical pathology-proven lesions and age-matched controls were
identified from the (censored) public mammography screening program. The primary outcome
was the AUC retrospectively classifying invasive breast cancer or ductal carcinoma in situ
(n=425) from no malignancy (n=490) or benign lesions (n=44) in age-matched controls. The
NYU system was tested in its original form, after training from scratch (without transfer
learning; TL), after retraining with TL, and both without (NYU1) and with (NYU2) heatmaps.

(heatmap examples)

#### Results
(table)
(AUCs)
The local test set comprised 959 clients. The original AUCs from the NYU1 and NYU2 models
were 0.83 and 0.89 respectively. The AUCs from the NYU1 and NYU2 models: 
- 1. applied in their original form to the local test set were 0.76 and 0.84 respectively; 
- 2. after local training without TL were 0.66 and 0.86 respectively; 
- 3. after retraining with TL were 0.82 and 0.86 respectively.

#### Conclusion 
A deep learning system developed on a US population showed reduced performance when applied ‘out of the box’ to an Australian population. The availability of model weights allowed local retraining with transfer learning which substantially improved the model performance. Provision of model weights for transfer learning may be beneficial to enable deep
learning systems to be adapted to local clinical environments.

#### Significance of Transfer Learning

Work by Wu et al. (3) included pre-training on over 100,000 clients, using Breast Imaging and Reporting Data System (BIRADS) categories as the target output. This pretraining ran for just under 2 weeks continuously (326 hours), on four Nvidia V100 GPUs. The Wu et al. exam-level CNN was trained for 12 hours on a Nvidia V100 GPU.(3) Benign and malignant heatmaps (see Figure 2), which in the current study improved performance, were generated as part of the Wu et al. freely available system,(6) including a model trained on 4,013 segmentations hand-drawn by NYU radiologists. The sharing of this time-consuming work was of considerable benefit in our experiments when building a "local" model, with retraining taking significantly less time than training from scratch on local data (see Main manuscript Table 2). We also note that retraining using the NYU weights was far quicker than the development of the original Wu et al. system, (6) despite using much less powerful hardware. In this way, the sharing of models, weights and code expedites research in the field. Recent works suggest that some combined modelling of fine-grained, lesion-level and larger-scale information is superior to modelling whole mammogram images alone. Strategies include using combinations of client-level, image-level and patch-level models(14), pretraining on lesion patches(15) and including patch-level heatmaps as input.(3,6) The importance of both this fine-grained information and the use of pretrained weights is highlighted in the drop in AUC observed with the NYU1 model trained from scratch, with randomly initialised weights.


## References

# Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening

**Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening**\
Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras\
2019

    @article{wu2019breastcancer, 
        title = {Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening},
        author = {Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, Stanis\l{}aw Jastrz\k{e}bski, Thibault F\'{e}vry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras}, 
        journal = {arXiv:1903.08297},
        year = {2019}
    }
