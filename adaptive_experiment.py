# сюда ставим фильтр zero-latency
# данные сохраняются тут
# применяется фильтр постоянно и определяет показ стимула



from scipy.signal import butter, lfilter
import numpy as np
import time
import cv2
import keyboard
import pickle
from  lsl_inlet import LSLInlet
#from filters import get_filtered
from kalman_experiment import lfilter_kf_alpha_beta

#streams = resolve_streams()


model_path = 'results/baseline_experiment_02-20_21-41-47_Vlad2/'




class Experiment:
    
        
         
    def receive_data(self):
        chunk, t_stamp = self.inlet.get_next_chunk()
        # print(f"{chunk=}")
        if chunk is not None:
            self.n_samples_in_chunk = len(chunk)
            print(self.n_samples_in_chunk)
            self.data[self.n_samples_received:self.n_samples_received + self.n_samples_in_chunk, :] = chunk
            
            
            
            chunk = self.ica_filter@chunk.T
           
            chunk, self.z  = lfilter(self.b,self.a,chunk, zi = self.z)
            chunk, self.z50  = lfilter(self.b50,self.a50,chunk, zi = self.z50)
            chunk, self.z100  = lfilter(self.b100,self.a100,chunk, zi = self.z100) 
            chunk, self.z200  = lfilter(self.b200,self.a200,chunk, zi = self.z200)
           
        
            
      
            self.n_samples_received += self.n_samples_in_chunk
            
            #self.filtered_kf_alpha = self.kf_alpha.get_filtered(chunk_alpha)
            #self.filtered_kf_beta = self.kf_beta.get_filtered(chunk_beta)
         
            self.envelope_kf_alpha,self.envelope_kf_beta = lfilter_kf_alpha_beta(self.kf, chunk)
            
            self.filtered_cfir_alpha = self.cfir_alpha.apply(chunk)
            self.filtered_cfir_beta = self.cfir_beta.apply(chunk)           
               
            self.envelope_cfir_alpha =np.abs(self.filtered_cfir_alpha)          
            self.envelope_cfir_beta =np.abs(self.filtered_cfir_beta)
            
            self.envelope_cfir_alpha = self.smoother_alpha_cfir.apply(self.envelope_cfir_alpha)
            self.envelope_cfir_beta = self.smoother_beta_cfir.apply(self.envelope_cfir_beta)
            
            
            
        else:
            self.n_samples_in_chunk =0
            self.filtered_kf_beta = None
            self.filtered_kf_alpha = None
            self.filtered_cfir_beta = None
            self.filtered_cfir_alpha = None
            
        return 
    #n_samples_in_chunk, filtered_kf_alpha, filtered_kf_beta, filtered_cfir_alpha, filtered_cfir_beta
              
            

    def main_process(self, model_path):
        
        time.sleep(1)
        
        file = open(model_path + 'model.pickle', "rb")
        
        
        
        dictionary = pickle.load(file)
        file.close()
        
        #high quntile alpha
        #low ...
        #... beta
        hqa_kf =dictionary['hqa_kf']
        lqa_kf =dictionary['lqa_kf']
        hqb_kf =dictionary['hqb_kf']
        lqb_kf =dictionary['lqb_kf']
        
        hqa_cfir =dictionary['hqa_cfir']
        lqa_cfir =dictionary['lqa_cfir']
        hqb_cfir =dictionary['hqb_cfir']
        lqb_cfir =dictionary['lqb_cfir']
        
        
        self.kf = dictionary['kf']

        
        self.cfir_alpha = dictionary['cfir_alpha']
        self.cfir_beta = dictionary['cfir_beta']
        
        self.b = dictionary['b']
        self.a = dictionary['a']
        self.b50 = dictionary['b50']
        self.a50 = dictionary['a50']
        self.b100 = dictionary['b100']
        self.a100 = dictionary['a100']
        self.b200 = dictionary['b200']
        self.a200 = dictionary['a200']
        
        
        self.ica_filter = dictionary['ica_filter']


        self.smoother_alpha_cfir = dictionary['smoother_alpha_cfir']
        self.smoother_beta_cfir = dictionary['smoother_beta_cfir']
        
        
        
        
        
        
        # ПО СТО НА КАЖДЫЙ ТИП, 4 периода по 200 пердъявлений
        
        seq = ['high_alpha_kf','low_alpha_kf', 'high_beta_kf','low_beta_kf','high_alpha_cfir','low_alpha_cfir', 'high_beta_cfir','low_beta_cfir',]*20
        multiseq = []
        for i in range(1):
            np.random.shuffle(seq)
            multiseq.append(seq)
        
        exp_settings = {
            
            
            
                    'exp_name': 'main_experiment',
                    'lsl_stream_name': 'UFO',
                    'blocks': {
                        'high_alpha_kf': {'id': 1},
                        'low_alpha_kf': {'id': 2},
                        'high_beta_kf': {'id': 3},
                        'low_beta_kf': {'id': 4},
                        'high_alpha_cfir': {'id': 5},
                        'low_alpha_cfir': {'id': 6},
                        'high_beta_cfir': {'id': 7},
                        'low_beta_cfir': {'id': 8}},
                    'sequences': multiseq,
                   
        
                    #максимальная буферизация принятых данных через lsl
                    'max_buflen': 5,  # in seconds!!!!
        
                    #максимальное число принятых семплов в чанке, после которых появляется возможность
                    #считать данные
                    'max_chunklen': 0,  # in number of samples!!!!
        }
        
        
        
        
        self.inlet = LSLInlet(exp_settings)
        self.inlet.srate = self.inlet.get_frequency()
        srate = int(round(self.inlet.srate))
        xml_info = self.inlet.info_as_xml()
        channel_names = self.inlet.get_channels_labels()
        
        n_channels = len(channel_names)
        
        
        cv2.namedWindow('go_image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('go_image', 1950, 1250)
        cv2.moveWindow('go_image', -1950, 200)
        
        red = cv2.imread("red.jpg")
        green = cv2.imread("green.jpg")
        white = cv2.imread("white.jpg")
        pause = cv2.imread("pause.jpg")
        

        # TWO HOURS OF TOTAL RECORDINGS
        self.n_samples_received = 0
        labels = np.zeros((800*10*srate))
        self.data = np.zeros((800*10*srate,n_channels))
        
        self.z = np.zeros(len(self.b)-1)
        self.z50 = np.zeros(len(self.b50)-1)
        self.z100 = np.zeros(len(self.b100)-1)
        self.z200 = np.zeros(len(self.b200)-1)
        
   
        
   
        
   
        for num_seq in range(len(exp_settings['sequences'])):
            
            cv2.imshow('go_image', pause)
            cv2.waitKey(1)
            while not keyboard.is_pressed('y'):
                #if (keyboard.read_key() != "y"):
                self.receive_data()
                #else:
                #    break
            cv2.imshow('go_image', red)
            cv2.waitKey(1) #??????
            print("You pressed y")
                
        
            
            
           
            for block_name in exp_settings['sequences'][num_seq]:
                trial_complete = False
           
                if block_name.find('low') != (-1):    
                    pause_dur = np.random.uniform(2.0,3.0)
                else:
                    pause_dur = np.random.uniform(5.0,10.0)
                n_samples_received_in_trial= 0
                
                
                
                while not trial_complete:
                    
                
                    self.receive_data()
                    if self.n_samples_in_chunk!=0:
                        n_samples_received_in_trial += self.n_samples_in_chunk
                        #print(self.n_samples_in_chunk)
                       
   
                        if  (n_samples_received_in_trial/srate)<pause_dur:
                            continue
                        
                        
                        
                            
                        if (((block_name == 'high_alpha_kf')and(self.envelope_kf_alpha[-1]>hqa_kf)) or              
                            ((block_name == 'low_alpha_kf')and(self.envelope_kf_alpha[-1]<lqa_kf)) or  
                            ((block_name == 'high_beta_kf')and(self.envelope_kf_beta[-1]>hqb_kf)) or
                            ((block_name == 'low_beta_kf')and(self.envelope_kf_beta[-1]<lqb_kf)) or
                            ((block_name == 'high_alpha_cfir')and(self.envelope_cfir_alpha[-1]>hqa_cfir)) or
                                ((block_name == 'low_alpha_cfir')and(self.envelope_cfir_alpha[-1]<lqa_cfir)) or
                                ((block_name == 'high_beta_cfir')and(self.envelope_cfir_beta[-1]>hqb_cfir)) or
                                ((block_name == 'low_beta_cfir')and(self.envelope_cfir_beta[-1]<lqb_cfir))):
                            
                            cv2.imshow('go_image', green)
                            cv2.waitKey(2) #??????
                            
                            #start_time
                            labels[self.n_samples_received]= exp_settings['blocks'][block_name]['id']
                            
                            
                            start_time= time.time()
                            while not keyboard.is_pressed('space'):
                                cur_time = time.time()
                                if ((cur_time-start_time) >5.0):
                                    self.receive_data()
                                    
                            #while keyboard.read_key() != "space":
                            #    pass
                            
                            # tight moment !!!!!!!!!!!!!!! may be timer better
                            
                            self.receive_data() 
                            labels[self.n_samples_received-1] = 111
                          
                            
                            cv2.imshow('go_image', red)
                            cv2.waitKey(2) #??????
                            print("You pressed Space")
                            trial_complete = True
        
                               
                
                
           
            
        
        self.data = self.data[:self.n_samples_received]
        labels = labels[:self.n_samples_received]
        
        
        file = open(model_path + 'experiment_results2.pickle', "wb")
        pickle.dump({'data': self.data,'labels':labels, 
                     'exp_settings': exp_settings}, file = file)
        file.close()
        
        
        end = cv2.imread("end.jpg")
        cv2.imshow('go_image', end)
        cv2.waitKey(2000)
                  
        
        
                 
        
                    
        
                
        
        
        
        

        
        
        
        
experiment = Experiment()
experiment.main_process(model_path)