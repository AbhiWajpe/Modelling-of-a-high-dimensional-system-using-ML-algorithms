# %%
import numpy as np

data_folder = '../bio_fsi/LilyPad/saved/'
data_file = 'pressure_.txt'
save_folder = ''

# %% get each time step of pressure readings as a string and put into a list
t_list = []
x_list = []
y_list = []
p_list = []

# read file line-by-line
# add line to respective list if found
# https://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list
idx = 0
with open(data_folder + data_file) as file:
    while line := file.readline():
        # print(line.rstrip())
        if idx == 1:
            x_list.append(line.rstrip())
            idx = 2
        elif idx == 2:
            y_list.append(line.rstrip())
            idx = 3
        elif idx == 3:
            p_list.append(line.rstrip())
            idx = 0
        elif 't = ' in line:
            # https://stackoverflow.com/questions/12572362/how-to-get-a-string-after-a-specific-substring
            t_list.append(line.rstrip().partition("t = ")[2])
            idx = 1

# make sure they are all the same length
assert len(t_list) == len(x_list)
assert len(t_list) == len(y_list)
assert len(t_list) == len(p_list)

# %% convert list of strings into array
N = len(p_list)
M = len(np.fromstring(p_list[0], sep=' '))

t_array = np.full(N, np.nan)
x_array = np.full((N, M), np.nan)
y_array = x_array.copy()
p_array = x_array.copy()

for n in range(N):
    t_array[n] = float(t_list[n])
    x_array[n, :] = np.fromstring(x_list[n], sep=' ')
    y_array[n, :] = np.fromstring(y_list[n], sep=' ')
    p_array[n, :] = np.fromstring(p_list[n], sep=' ')

# %% save
np.save(save_folder + 't.npy', t_array)
np.save(save_folder + 'x.npy', x_array)
np.save(save_folder + 'y.npy', y_array)
np.save(save_folder + 'p.npy', p_array)

# %%
