Out-of-the-box options
======================

.. |br| raw:: html

  <br/>

Models
------

.. list-table::
   :header-rows: 1

   * - Module config key
     - Available options
   * - ``model``
     - ``timm_<model_name>`` |br| ``timm_pretrained_<model_name>`` |br| ``torchvision_<model_name>`` |br| ``torchvision_pretrained_<model_name>``

Evaluation metrics
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module config key
     - Available options
   * - ``evaluations``
     - ``adversarial_accuracy`` |br| ``auto_attack`` |br| ``classification_accuracy`` |br| ``classification_accuracy_per_group`` |br| ``demographic_parity_inferred_groups`` |br| ``expected_calibration_error`` |br| ``foolbox_attacks`` |br| ``l2_auto_attack`` |br| ``l2_auto_attacks`` |br| ``l2_custom_auto_attack`` |br| ``l2_custom_auto_attacks`` |br| ``linf_apgd_ce_auto_attack`` |br| ``linf_apgd_ce_auto_attacks`` |br| ``linf_apgd_dlr_auto_attack`` |br| ``linf_apgd_dlr_auto_attacks`` |br| ``linf_auto_attack`` |br| ``linf_auto_attacks`` |br| ``linf_custom_auto_attack`` |br| ``linf_custom_auto_attacks`` |br| ``mock_evaluation`` |br| ``negative_log_likelihood`` |br| ``number_of_parameters`` |br| ``output_distribution_per_group`` |br| ``top_k_classificaton_accuracy``

Corruptions
-----------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module config key
     - Available options
   * - ``corruption``
     - ``no_corruption`` |br| ``imagenet_c_type``
   * - ``corruption.corruption_types``
     - ``gaussian_noise`` |br| ``shot_noise`` |br| ``impulse_noise`` |br| ``defocus_blur`` |br| ``glass_blur`` |br| ``motion_blur`` |br| ``zoom_blur`` |br| ``snow`` |br| ``fog`` |br| ``brightness`` |br| ``contrast`` |br| ``elastic_transform`` |br| ``pixelate`` |br| ``jpeg_compression`` |br| ``speckle_noise`` |br| ``gaussian_blur`` |br| ``spatter`` |br| ``saturate``

Transformations
---------------

Datasets
--------

.. csv-table::
   :file: registered_datasets.csv
   :widths: 30 70
   :header-rows: 1
