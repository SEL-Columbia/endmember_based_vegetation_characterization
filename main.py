import time, os, yaml, argparse
from spectral_unmixing import *
from dataloader import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from plotting import *

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

    interpolate_rainfall(args)

    if args.spectral_unmixing:
        print('Load EVI Image')

        with rasterio.open(os.path.join(args.base_dir, 'imagery', 'modis',
                                        args.evi_img_filename.format(args.unmixing_region))) as src:
            evi_img = src.read()
            img_meta = src.meta

        save_file_endmembers = os.path.join(args.base_dir, 'saved_endmembers', args.unmixing_region,
                                            'extracted_endmembers_{}_outphasetype_{}_nclusters_{}_nsamples_{}.csv'.format(
                                                args.unmixing_region, args.outphase_endmember_type,
                                                args.num_clusters, args.num_samples))
        if not args.load_existing_endmembers:
            print('Saving New Endmembers')
            generate_endmembers(args, evi_img, save_file_endmembers)

        print('Loading Endmembers')
        endmember_array = np.array(pd.read_csv(save_file_endmembers, index_col=0, header=0))

        if args.plotting_endmembers:
            rainfall_ts = interpolate_rainfall(args)
            plot_endmembers(endmember_array, rainfall_ts, args.unmixing_region)

        if args.calc_new_abundance_map:
            print('Calculating and Saving Abundance Map')
            spectral_unmixing_main(args, evi_img, img_meta, endmember_array, args.unmixing_method)

    if args.irrig_prediction:
        print('Predicting Irrigation')
        dataloader = DataGenerator(args)
        print('Loading Data')
        X_train, X_val, y_train, y_val = dataloader.return_data()


        forest = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=2, verbose=True,
                                        n_jobs=3, class_weight='balanced')




        forest.fit(X_train, y_train)
        print(forest.feature_importances_)


        print("Accuracy on training set, 1: {:.3f}".format(forest.score(X_train, y_train)))
        print("Accuracy on validation set, 1: {:.3f}".format(forest.score(X_val, y_val)))


        y_val_predicts = forest.predict(X_val)
        cf_matrix_val = confusion_matrix(y_val, y_val_predicts)

        print("Confusion matrix for validation set")
        print(cf_matrix_val)
