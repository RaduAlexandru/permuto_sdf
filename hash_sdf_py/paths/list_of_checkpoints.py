#just a storage list of the checkpoints for DTU that we use 
#it's then easy from another file to do import list_of_checkpoints and use list_chkps.dtu[scene_nr]
ckpts={

    #DTU
    "dtu_with_mask_True":{
        "dtu_scan24": "full_dtu_dtu_scan24_with_mask_True_test/5000",
        "dtu_scan37": "full_dtu_dtu_scan37_with_mask_True_test/5000",
        "dtu_scan40": "full_dtu_dtu_scan40_with_mask_True_test/5000",
        "dtu_scan55": "full_dtu_dtu_scan55_with_mask_True_test/5000",
        "dtu_scan63": "full_dtu_dtu_scan63_with_mask_True_test/5000",
        "dtu_scan65": "full_dtu_dtu_scan65_with_mask_True_test/5000",
        "dtu_scan69": "full_dtu_dtu_scan69_with_mask_True_test/5000",
        "dtu_scan83": "full_dtu_dtu_scan83_with_mask_True_test/5000",
        "dtu_scan97": "full_dtu_dtu_scan97_with_mask_True_test/5000",
        "dtu_scan105": "full_dtu_dtu_scan105_with_mask_True_test/5000",
        "dtu_scan106": "full_dtu_dtu_scan106_with_mask_True_test/5000",
        "dtu_scan110": "full_dtu_dtu_scan110_with_mask_True_test/5000",
        "dtu_scan114": "full_dtu_dtu_scan114_with_mask_True_test/5000",
        "dtu_scan118": "full_dtu_dtu_scan118_with_mask_True_test/5000",
        "dtu_scan122": "full_dtu_dtu_scan122_with_mask_True_test/5000"
    },
    


    #multiface
    "multiface_without_mask_True_use_all_imgs_False":{
        #v2
        # "path_prefix_home": "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/batch_training_v2/multiface/without_mask_True_use_all_imgs_False/checkpoints",
        # "path_prefix_remote": "/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/batch_training_v2/multiface/without_mask_True_use_all_imgs_False/checkpoints",
        # v6 (fixed the lipshitz loss being always zero and reduced eikw to0.04)
        # "path_prefix_home": "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/batch_training_v6_Fix_eik0.04eikDecay0.01/multiface/without_mask_True_use_all_imgs_False/checkpoints",
        # "path_prefix_remote": "/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/batch_training_v6_Fix_eik0.04eikDecay0.01/multiface/without_mask_True_use_all_imgs_False/checkpoints",
        # v7 No lipshitz, just WD
        "path_prefix_home": "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/batch_training_v7_NoLipshitzJustWD/multiface/without_mask_True_use_all_imgs_False/checkpoints",
        "path_prefix_remote": "/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/batch_training_v7_NoLipshitzJustWD/multiface/without_mask_True_use_all_imgs_False/checkpoints",

        "0": "r_batchTrain_v7_NoLipshitzJustWD_multiface_0_without_mask_True_use_all_imgs_False/200000",
        "1": "r_batchTrain_v7_NoLipshitzJustWD_multiface_1_without_mask_True_use_all_imgs_False/200000",
        "2": "r_batchTrain_v7_NoLipshitzJustWD_multiface_2_without_mask_True_use_all_imgs_False/200000",
        "3": "r_batchTrain_v7_NoLipshitzJustWD_multiface_3_without_mask_True_use_all_imgs_False/200000",
        "4": "r_batchTrain_v7_NoLipshitzJustWD_multiface_4_without_mask_True_use_all_imgs_False/200000",
        "5": "r_batchTrain_v7_NoLipshitzJustWD_multiface_5_without_mask_True_use_all_imgs_False/200000",
        "6": "r_batchTrain_v7_NoLipshitzJustWD_multiface_6_without_mask_True_use_all_imgs_False/200000",
        "7": "r_batchTrain_v7_NoLipshitzJustWD_multiface_7_without_mask_True_use_all_imgs_False/200000"
        
    },





    #INGP
    #DTU
    "dtu_without_mask_False_use_all_imgs_False_ingp":{
        # v7 No lipshitz, just WD
        "path_prefix_home": "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/ingp/dtu/without_mask_False_use_all_imgs_False/checkpoints",
        "path_prefix_remote": "/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/ingp/dtu/without_mask_False_use_all_imgs_False/checkpoints",

        "dtu_scan24": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan24_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan37": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan37_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan40": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan40_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan55": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan55_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan63": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan63_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan65": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan65_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan69": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan69_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan83": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan83_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan97": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan97_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan105": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan105_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan106": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan106_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan110": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan110_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan114": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan114_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan118": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan118_without_mask_False_use_all_imgs_False_ingp/100000",
        "dtu_scan122": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan122_without_mask_False_use_all_imgs_False_ingp/100000"
        
    },
    "dtu_without_mask_True_use_all_imgs_False_ingp":{
        # v7 No lipshitz, just WD
        "path_prefix_home": "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/ingp/dtu/without_mask_True_use_all_imgs_False/checkpoints",
        "path_prefix_remote": "/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/ingp/dtu/without_mask_True_use_all_imgs_False/checkpoints",

        "dtu_scan24": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan24_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan37": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan37_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan40": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan40_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan55": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan55_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan63": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan63_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan65": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan65_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan69": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan69_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan83": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan83_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan97": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan97_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan105": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan105_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan106": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan106_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan110": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan110_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan114": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan114_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan118": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan118_without_mask_True_use_all_imgs_False_ingp/100000",
        "dtu_scan122": "r_batchTrain_v7_NoLipshitzJustWD_dtu_dtu_scan122_without_mask_True_use_all_imgs_False_ingp/100000"
    },





    #NEUS
    #MULTIFACE
    "multiface_without_mask_True_use_all_imgs_False_neus":{
        # v7 No lipshitz, just WD
        "path_prefix_home": "/media/rosu/Data/phd/c_ws/src/phenorob/instant_ngp_2/checkpoints/neus/multiface/without_mask_True_use_all_imgs_False/checkpoints",
        "path_prefix_remote": "/home/user/rosu/c_ws/src/instant_ngp_2/checkpoints/neus/multiface/without_mask_True_use_all_imgs_False/checkpoints",

        "0": "r_batchTrain_v7_NoLipshitzJustWD_multiface_0_without_mask_True_use_all_imgs_False_neus/300000",
        "1": "r_batchTrain_v7_NoLipshitzJustWD_multiface_1_without_mask_True_use_all_imgs_False_neus/300000",
        "2": "r_batchTrain_v7_NoLipshitzJustWD_multiface_2_without_mask_True_use_all_imgs_False_neus/300000",
        "3": "r_batchTrain_v7_NoLipshitzJustWD_multiface_3_without_mask_True_use_all_imgs_False_neus/300000",
        "4": "r_batchTrain_v7_NoLipshitzJustWD_multiface_4_without_mask_True_use_all_imgs_False_neus/300000",
        "5": "r_batchTrain_v7_NoLipshitzJustWD_multiface_5_without_mask_True_use_all_imgs_False_neus/300000",
        "6": "r_batchTrain_v7_NoLipshitzJustWD_multiface_6_without_mask_True_use_all_imgs_False_neus/300000",
        "7": "r_batchTrain_v7_NoLipshitzJustWD_multiface_7_without_mask_True_use_all_imgs_False_neus/300000"
        
    }, 

   

}