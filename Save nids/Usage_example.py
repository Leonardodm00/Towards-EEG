base_path = '/content/drive/MyDrive/Colab Notebooks/Aligned Neurons HOC'
nids = [
    
        
        xxxx,yyyy,zzzz, ...



]

# Available prefix (from params_evoked_with_EEG)
# self.y_in_Y = [
#                 [['p23'],['b23','nb23']],
#                 [['p4','ss4(L23)','ss4(L4)'],['b4','nb4']],
#                 [['p5(L23)','p5(L56)'],['b5','nb5']],
#                 [['p6(L4)','p6(L56)'],['b6','nb6']]]



prefix = 'p23'



save_nids(nids, base_path, prefix, extension=".txt")
