import librosa
from scipy.io.wavfile import write

def audio_preprocessing(file_name, input_path, output_path):

    audio_path = input_path + file_name
    y, sr = librosa.load(audio_path)

    y_normed = (y/max(max(y), abs(min(y))))* 0.02 # нормализация по максимальному значению, домножаем на 0,02, чтобы не возникало слишком громкого звука
    y_filtered_norm = librosa.effects.preemphasis(y_normed) # фильтром сглаживаем нормализованную аудиодорожку, чтобы перепады были менее выраженными
    max_y = max(max(y_filtered_norm), abs(min(y_filtered_norm))) # определяем максимальное значение амплитуды по всей аудиодорожке (отфильтр.+ нормирован.)

    first_bigger = 0
    for num, i in enumerate(y_filtered_norm): # в цикле определяем искодную точку, с которой будет начинаться обрезанная аудиодорожка (порядковый номер, амплитуда)
        if abs(i) > max_y*0.2: # сравниваем амплитуду по подулю, если больше, чем ~20% от максималной амплитуды во всей дорожке, то эту амплитуда = исходная точка
            first_bigger = num
            break

    y_filtered_norm_cut = y_filtered_norm[first_bigger:first_bigger + sr] # обрезаем дорожку, начиная с исходной точки до исх.т.+частота дискрет. (1 секунда)
    output_audio_path = output_path + file_name
    write(output_audio_path, sr, y_filtered_norm_cut) # записываем путь до предобработанной дорожки

    return y_filtered_norm_cut

