## Credit risk prediction and explainability with Amazon SageMaker

We first build an ensemble model that takes tabular input data and produces multiple prediction outcomes for each individual model and the combined ensemble outcome. 

The input features include financial information of the requestor like checking account status, credit history, number of credits, and loan information like amount, duration, purpose, installment rate and also demographic information like personal status and sex combination, age, foreign worker status. Total number of features are 20. The ensemble model produces a probability of good credit (low risk) for each individual and combined model. 

We then analyze this black box model and input data for bias detection and explainability with SageMaker Clarify. The output is aggregate and individual SHAP values for each input record and a summary of bias metrics for data and model. 

The attached notebook can be run in Amazon SageMaker Studio. 

<img width="982" alt="Screenshot 2021-08-10 at 13 21 39" src="https://user-images.githubusercontent.com/8871432/128865871-6603cab8-e705-4d93-ba68-4e5e7d2a0da4.png">


## License

This library is licensed under the MIT-0 License. See the LICENSE file.

