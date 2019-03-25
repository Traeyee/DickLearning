import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    # run options
    parser.add_argument('--run_type', default="continue", help="new/continue/finetune")
    parser.add_argument('--task_type')
    parser.add_argument('--use_profile', default=0, type=int)
    parser.add_argument('--logdir', default="log/0", help="log directory")  # crucial

    # vocabulary
    parser.add_argument('--use_auto_vocab', default=0, type=int)
    parser.add_argument('--vocab', help="vocabulary file path")
    parser.add_argument('--vocabs', help="vocabulary file path. New interface, used for multiple inputs")

    # train
    # files
    parser.add_argument('--train_data')
    parser.add_argument('--eval_data')
    parser.add_argument('--train1', help="german training segmented data")
    parser.add_argument('--train2', default='./couplet/train/out.10000',
                        help="english training segmented data")
    parser.add_argument('--eval1', default='./couplet/train/in.100',
                        help="german evaluation segmented data")
    parser.add_argument('--eval2', default='./couplet/train/out.100',
                        help="english evaluation segmented data")
    parser.add_argument('--test', default='./couplet/train/out.100',
                        help="english evaluation unsegmented data")
    parser.add_argument('--pb_name', default='default')

    # training scheme
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default=64, type=int)

    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/0", help="evaluation dir")

    parser.add_argument('--zero_step', default=0, type=int, help="start from global_step=0")

    # model
    parser.add_argument('--inputs', help="input indices of the model")
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test1', default='iwslt2016/segmented/test.de.bpe',
                        help="german test segmented data")
    parser.add_argument('--test2', default='iwslt2016/prepro/test.en',
                        help="english test data")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=64, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")
