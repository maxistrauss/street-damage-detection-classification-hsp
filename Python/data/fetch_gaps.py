from gaps_dataset import gaps

gaps.download(login='gapsro2s;i2A*7',
                output_dir='Python/data/GAPsV2',
                version=2,
                patchsize=160,
                issue='NORMvsDISTRESS_50k',
                debug_outputs=True)

x_train0, y_train0 = gaps.load_chunk(chunk_id=0,
                                           version=2,
                                           patchsize=160,
                                           issue='NORMvsDISTRESS_50k',
                                           subset='train',
                                           datadir='Python/data/GAPsV2') #load the first chunk of the dataset

print(x_train0)
print(y_train0)

# gaps.download(login='replace_with_your_login',
#                 output_dir='desired folder',
#                 version='10m',
#                 patchsize='segmentation',
#                 issue='ASFaLT')