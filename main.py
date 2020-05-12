import time, os, yaml, argparse
from spectral_unmixing import *
from dataloader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from plotting import *
from chirps_processing import *

# Load in parameters from params.yaml
def get_args():
    parser = argparse.ArgumentParser(
        description= 'Predict irrigation presence from abundance maps')

    parser.add_argument('--training_params_filename',
                        type=str,
                        default='params.yaml',
                        help='Filename defining model configuration')

    args = parser.parse_args()
    config = yaml.load(open(args.training_params_filename))
    for k, v in config.items():
        args.__dict__[k] = v


    return args

if __name__ == '__main__':


    args = get_args()

    # Cluster rainfall timeseries and create rainfall region shapefiles
    if args.rainfall_cluster:
        cluster_rainfall(args)

    # Plot rainfall region timeseries and shapefiles
    if args.rainfall_cluster_plotting:
        rainfall_region_plotting(args)


    # Endmember extraction and abundance map creation
    if args.spectral_unmixing:
        print('Load EVI Image')

        # Load image
        img_src =  rasterio.open(os.path.join(args.base_dir, 'imagery', 'modis',
                            args.evi_img_filename.format(args.unmixing_region)))

        print('Loading Endmembers')
        endmember_array = return_endmembers(args, img_src)


        if args.plotting_endmembers:
            plot_endmembers(args, endmember_array)

        if args.calc_new_abundance_map:
            print('Calculating and Saving Abundance Map')
            spectral_unmixing_main(args, img_src, endmember_array, args.unmixing_method)




    if args.irrig_prediction:
        print('Predicting Irrigation')
        dataloader = DataGenerator(args)
        print('Loading Data')
        X_train, X_val, y_train, y_val = dataloader.return_data()

        # Create classifier
        forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=2, verbose=True,
                                        n_jobs=3, class_weight='balanced')

        print('Fit classifier')
        forest.fit(X_train, y_train)
        print('Importance of features for prediction: {}'.format(forest.feature_importances_))

        print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
        print("Accuracy on validation set: {:.3f}".format(forest.score(X_val, y_val)))

        y_val_predicts = forest.predict(X_val)
        cf_matrix_val = confusion_matrix(y_val, y_val_predicts)

        print("Confusion matrix for validation set")
        print(cf_matrix_val)
