import argparse

if __name__ == '__main__':
    # Setup arguments.
    # 
    # You can use the arguments like this:
    # print(args.traning)
    # print(args.output)
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # Load training data.

    # Train the model.

    # Predict the result. 

    # Output the result to csv file.
