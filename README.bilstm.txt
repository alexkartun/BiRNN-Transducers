How to run bilstmTrain.py file?

Type:
    python3 bilstmTrain.py repr train_filename, model_filename, type_of_data_set

Should be 4 parameters to the script.
1) repr - is one of a, b, c, d representations.
2) train_filename - train's filename.
3) model_filename - model's filename.
4) type_of_data_set - name of the directory where the files are located

   F.E: python3 bilstmTrain.py a train model pos
   * a: is the word representation
   * 'pos': directory that located near the experiment.py file
   * train: file located in the 'pos' directory
   * model: name of the zip file which will include all the trained model's data,
     it will be saved in the 'pos' directory.


How to run bilstmTag.py file?

Type:
    python3 bilstmTag.py repr model_filename test_filename type_of_data_set

Should be 4 parameters to the script.
1) repr - is one of a, b, c, d representations.
2) model_filename - model's filename.
3) test_filename - test's filename.
4) type_of_data_set - name of the directory where the files are located

   F.E: python3 bilstmTag.py a model test pos
   * a: is the word representation
   * 'pos': directory that located near the experiment.py file
   * test: file located in the 'pos' directory
   * model: name of the zip file which include all the trained model's data,
     it located in the 'pos' directory.
    Note: the output of this script will be saved in 'pos' directory and his name will be as 'a' as representation type.