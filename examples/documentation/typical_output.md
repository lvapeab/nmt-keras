# NMT-Keras' output 

This document shows and explains the typical output produced by the training pipeline of NMT-Keras.
 
Assuming that we launched NMT-Keras for the example from tutorials, we'll have the following tree of folders (after 1 epoch):

```bash 
├── trained_models
│   ├── EuTrans_GroundHogModel_src_emb_420_bidir_True_enc_600_dec_600_deepout_maxout_trg_emb_420_Adam_0.001
│   │   ├── config.pkl
│   │   ├── epoch_1_Model_Wrapper.pkl
│   │   ├── epoch_1_structure_init.json
│   │   ├── epoch_1_structure.json
│   │   ├── epoch_1_structure_next.json
│   │   ├── epoch_1_weights.h5
│   │   ├── epoch_1_weights_init.h5
│   │   ├── epoch_1_weights_next.h5
│   │   ├── val.coco
│   │   ├── val_epoch_1.pred
```

Let's have a look to these files.

* `config.pkl`: Pickle containing the training parameters.
    
* `epoch_1_Model_Wrapper.pkl`: Pickle containing the Model_Wrapper object that we have trained.

* `epoch_1_structure.json`:  Keras json specifying the layer and connections of the model.

* `epoch_1_structure_init.json`: Keras json specifying the layer and connections of the model_init (see tutorial 4 for more info about the model).
    
* `epoch_1_structure_next.json`: Keras json specifying the layer and connections of the model_next (see tutorial 4 for more info about the model).
    
* `epoch_1_weights.h5`: Model parameters (weight matrices).
    
* `epoch_1_weights_init.h5`: Model init parameters (weight matrices).
    
* `epoch_1_weights_next.h5`: Model next parameters (weight matrices).
    
* `val.coco`: Metrics dump. This file is name as [tested_split].[metrics_name]. It contains a header with the metrics name and the value of all evaluations (epoch/updates). For instance:
     
    ```
    epoch,Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR, ROUGE_L, 
    1,0.906982874122, 0.875873151361, 0.850383597611, 0.824070996966, 8.084477458, 0.550547408997, 0.931523374569, 
    2,0.932937494321, 0.90923787501, 0.889965151506, 0.871819102335, 8.53565391657, 0.586377788443, 0.947634196936, 
    3,0.965579088172, 0.947927460597, 0.934090548706, 0.920166838768, 9.0864109399, 0.63234570058, 0.971618921459, 
     ```
* `val_epoch_1.pred`: Raw file with the output of the NMT system at the evaluation performed at the end of epoch 1.
    
    
We can modify the save and evaluation frequencies from the `config.py` file.  
    