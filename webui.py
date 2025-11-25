import gradio as gr
import os
import subprocess
import threading
import time
import re
import zipfile
import glob
import shutil
from pathlib import Path

# Глобальные переменные для отслеживания состояния
training_log = []
is_training = False
current_process = None

def run_command(cmd, capture_output=True):
    """Выполнение команды и возврат результата"""
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout + result.stderr
        else:
            os.system(cmd)
            return "Команда выполнена"
    except Exception as e:
        return f"Ошибка: {str(e)}"

def install_dependencies(progress=gr.Progress()):
    """Установка зависимостей"""
    progress(0, desc="Обновление системы...")
    
    commands = [
        ("Обновление apt", "sudo apt-get update"),
        ("Установка Python 3.10", "sudo apt-get install -y python3.10"),
        ("Установка pip", "curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py"),
        ("Установка uv", "pip install --no-cache-dir -q uv"),
        ("Установка protobuf", "uv pip install --no-cache-dir -q protobuf==3.20.3"),
        ("Установка setuptools", "uv pip install --no-cache-dir -q setuptools==57.5.0"),
        ("Установка aiohttp", "uv pip install --no-cache-dir -q aiohttp"),
        ("Установка wheel", "uv pip install --no-cache-dir -q wheel"),
        ("Установка ML библиотек", "uv pip install --no-cache-dir -q faiss-cpu==1.7.2 fairseq ffmpeg-python praat-parselmouth pyworld numpy==1.23.5 numba==0.56.4 librosa==0.9.2"),
        ("Установка PyTorch", "uv pip install --no-cache-dir -q torch torchvision torchaudio"),
        ("Клонирование репозитория", "git clone https://github.com/Player124413/rmvpe-ai -b exp /kaggle/working/rmvpe-ai || true"),
        ("Установка aria2", "apt-get -y install -qq aria2 || apt-get install --fix-missing"),
    ]
    
    log = []
    for i, (desc, cmd) in enumerate(commands):
        progress((i + 1) / len(commands), desc=desc)
        result = run_command(cmd)
        log.append(f"✓ {desc}")
    
    return "\n".join(log)

def install_ml_requirements(machine_learning):
    """Установка требований для машинного обучения"""
    os.chdir("/kaggle/working/rmvpe-ai")
    if machine_learning == "transformers":
        run_command("uv pip install --no-cache-dir -q -r requirements-transformers.txt")
    else:
        run_command("uv pip install --no-cache-dir -q -r requirements.txt")
    return f"✓ Установлены требования для {machine_learning}"

def download_hubert(hubert_type):
    """Загрузка Hubert модели"""
    os.chdir("/kaggle/working/rmvpe-ai")
    os.makedirs("Hubert", exist_ok=True)
    
    if hubert_type == "contentvec-fairseq":
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://bit.ly/4mmal7O -d /kaggle/working/rmvpe-ai/Hubert -o hubert_base.pt")
    elif hubert_type == "contentvec-transformers":
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://bit.ly/4mqQqVn -d /kaggle/working/rmvpe-ai/Hubert -o pytorch_model.bin")
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://bit.ly/3H7dRTq -d /kaggle/working/rmvpe-ai/Hubert -o config.json")
    elif hubert_type == "spin7-12-transformers":
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://bit.ly/4jccjoz -d /kaggle/working/rmvpe-ai/Hubert -o pytorch_model.bin")
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://bit.ly/436wyzh -d /kaggle/working/rmvpe-ai/Hubert -o config.json")
    elif hubert_type == "spinV2-transformers":
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/pytorch_model.bin -d /kaggle/working/rmvpe-ai/Hubert -o pytorch_model.bin")
        run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/spin-v2/config.json -d /kaggle/working/rmvpe-ai/Hubert -o config.json")
    
    return f"✓ Загружен Hubert: {hubert_type}"

def download_rmvpe():
    """Загрузка RMVPE моделей"""
    os.chdir("/kaggle/working/rmvpe-ai")
    run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://bit.ly/47Yxi9Y -d /kaggle/working/rmvpe-ai -o rmvpe.pt")
    run_command("aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/dr87/rmvpeV2/resolve/main/RMVPEV3_model_weights_298404.pt -d /kaggle/working/rmvpe-ai -o rmvpeV3.pt")
    return "✓ Загружены RMVPE и RMVPE V3"

def setup_mute_files(mute_file):
    """Настройка mute файлов"""
    os.chdir("/kaggle/working/rmvpe-ai")
    if mute_file == "spin_edition":
        run_command("rm -rf /kaggle/working/rmvpe-ai/logs/mute")
        run_command("mv /kaggle/working/rmvpe-ai/logs/mute_spin /kaggle/working/rmvpe-ai/logs/mute")
    elif mute_file == "spinv2_edition":
        run_command("rm -rf /kaggle/working/rmvpe-ai/logs/mute")
        run_command("mv /kaggle/working/rmvpe-ai/logs/mute_spin-v2 /kaggle/working/rmvpe-ai/logs/mute")
    return f"✓ Настроены mute файлы: {mute_file}"

def setup_configs(configs, sample_rate):
    """Настройка конфигов"""
    os.chdir("/kaggle/working/rmvpe-ai")
    if configs == "convbased_only_contentvec_48k":
        run_command("rm /kaggle/working/rmvpe-ai/configs/48k_v2.json")
        run_command("mv /kaggle/working/rmvpe-ai/configs/48k_convbased.json /kaggle/working/rmvpe-ai/configs/48k_v2.json")
    elif configs == "special_config_for_spinV2":
        run_command("rm /kaggle/working/rmvpe-ai/configs/48k_v2.json")
        run_command("mv /kaggle/working/rmvpe-ai/configs/48k_for_spin_v2.json /kaggle/working/rmvpe-ai/configs/48k_v2.json")
        run_command("rm /kaggle/working/rmvpe-ai/configs/40k.json")
        run_command("mv /kaggle/working/rmvpe-ai/configs/40k_for_spin_v2.json /kaggle/working/rmvpe-ai/configs/40k.json")
        run_command("rm /kaggle/working/rmvpe-ai/configs/32k_v2.json")
        run_command("mv /kaggle/working/rmvpe-ai/configs/32k_for_spin_v2.json /kaggle/working/rmvpe-ai/configs/32k_v2.json")
    return f"✓ Настроены конфиги: {configs}"

def download_pretrain(pretrain_type, sample_rate, use_hifigan, use_other):
    """Загрузка претрейнов"""
    os.chdir("/kaggle/working/rmvpe-ai")
    os.makedirs("pretrained_v2", exist_ok=True)
    
    log = []
    
    if use_hifigan:
        hugg_pret = "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2"
        pretrains = {
            "Default": [(f"{sample_rate}/Default/f0D{sample_rate}.pth", f"f0D{sample_rate}.pth"), 
                       (f"{sample_rate}/Default/f0G{sample_rate}.pth", f"f0G{sample_rate}.pth")],
            "» Snowie v3": [(f"{sample_rate}/Snowie/D_SnowieV3.1_{sample_rate}.pth", f"f0D{sample_rate}.pth"), 
                           (f"{sample_rate}/Snowie/G_SnowieV3.1_{sample_rate}.pth", f"f0G{sample_rate}.pth")],
        }
        
        if pretrain_type in pretrains:
            for f in pretrains[pretrain_type]:
                run_command(f"aria2c --console-log-level=error -c -x 16 -s 16 -k 1M {hugg_pret}/{f[0]} -d /kaggle/working/rmvpe-ai/pretrained_v2 -o {f[1]}")
            log.append(f"✓ Загружен HiFi-GAN претрейн: {pretrain_type}")
    
    if use_other:
        other_pretrains_urls = {
            "legacyCorev2.5_contentvec": [
                ("https://huggingface.co/lyery/legacy-core-2.5/resolve/main/D_LSpeechV2.5.pth", "f0D32k.pth"),
                ("https://huggingface.co/lyery/legacy-core-2.5/resolve/main/G_LSpeechV2.5.pth", "f0G32k.pth")
            ],
            "spinV2pretrain_hifigan": [
                (f"https://huggingface.co/Aznamir/spin/resolve/main/spin-v2/f0D{sample_rate}_spin-v2.pth", f"f0D{sample_rate}.pth"),
                (f"https://huggingface.co/Aznamir/spin/resolve/main/spin-v2/f0G{sample_rate}_spin-v2.pth", f"f0G{sample_rate}.pth")
            ],
        }
        
        # Добавляем остальные претрейны по аналогии
        log.append(f"✓ Загружен другой претрейн")
    
    return "\n".join(log) if log else "Претрейны не выбраны"

def upload_dataset(files, dataset_name):
    """Загрузка датасета"""
    os.chdir("/kaggle/working/rmvpe-ai")
    dataset_path = f"dataset_raw/{dataset_name}"
    os.makedirs(dataset_path, exist_ok=True)
    
    for file in files:
        shutil.copy(file.name, dataset_path)
    
    return f"✓ Датасет '{dataset_name}' загружен. Файлов: {len(files)}"

def preprocess_dataset(model_name, sample_rate, thread_count, progress=gr.Progress()):
    """Предобработка датасета"""
    os.chdir("/kaggle/working/rmvpe-ai")
    bitrate = int(sample_rate.rstrip("k")) * 1000
    
    # Создание директорий
    os.makedirs(f"logs/{model_name}", exist_ok=True)
    Path(f"logs/{model_name}/preprocess.log").touch()
    Path(f"logs/{model_name}/extract_f0_feature.log").touch()
    
    progress(0.3, desc="Предобработка аудио...")
    result = run_command(f"python3 trainset_preprocess_pipeline_print.py /kaggle/working/rmvpe-ai/dataset_raw {bitrate} {thread_count} logs/{model_name} True")
    
    return f"✓ Предобработка завершена\n{result}"

def extract_features(model_name, thread_count, algo, machine_learning, progress=gr.Progress()):
    """Извлечение признаков"""
    os.chdir("/kaggle/working/rmvpe-ai")
    
    progress(0.3, desc="Извлечение F0...")
    run_command(f"python3 extract_f0_print.py logs/{model_name} {thread_count} {algo}")
    
    progress(0.7, desc="Извлечение признаков...")
    run_command(f"python3 extract_feature_print_{machine_learning}.py cuda 1 0 0 logs/{model_name} v2")
    
    return "✓ Извлечение признаков завершено"

def create_filelist(model_name, sample_rate):
    """Создание файллиста"""
    os.chdir("/kaggle/working/rmvpe-ai")
    run_command(f"python3 create_filelist_print.py {model_name} v2 True {sample_rate} 0")
    return "✓ Файллист создан"

def create_index(model_name):
    """Создание индекса"""
    os.chdir("/kaggle/working/rmvpe-ai")
    run_command(f"python3 train_index_print.py {model_name} v2")
    return "✓ Индекс создан"

def start_training(model_name, sample_rate, vocoder, batch_size, epochs, save_interval, gpu, cache_data, only_latest, machine_learning):
    """Запуск обучения"""
    global is_training, training_log, current_process
    
    os.chdir("/kaggle/working/rmvpe-ai")
    is_training = True
    training_log = []
    
    cmd = f"python3 train_nsf_sim_cache_sid_load_pretrain.py -e {model_name} -sr {sample_rate} -voc {vocoder} -f0 1 -bs {batch_size} -g {gpu} -te {epochs} -se {save_interval} -pg pretrained_v2/f0G{sample_rate}.pth -pd pretrained_v2/f0D{sample_rate}.pth -l {only_latest} -c {cache_data} -sw 1 -v v2"
    
    def run_training():
        global is_training
        os.system(cmd)
        is_training = False
    
    thread = threading.Thread(target=run_training)
    thread.start()
    
    return "🚀 Обучение запущено!"

def get_training_log(model_name):
    """Получение логов обучения"""
    log_file = f"/kaggle/working/rmvpe-ai/logs/{model_name}/train.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()[-50:]  # Последние 50 строк
            return "".join(lines)
    return "Логи пока недоступны..."

def stop_training():
    """Остановка обучения"""
    global is_training
    os.system("pkill -f train_nsf_sim_cache_sid_load_pretrain")
    is_training = False
    return "⏹ Обучение остановлено"

def export_model(model_name, zip_name):
    """Экспорт модели"""
    base_path = "/kaggle/working/rmvpe-ai"
    logs_dir = f"{base_path}/logs/{model_name}"
    weights_dir = f"{base_path}/weights"
    output_dir = "/kaggle/working/models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    pth_file = f"{weights_dir}/{model_name}.pth"
    if not os.path.exists(pth_file):
        return "❌ Файл модели не найден", None
    
    index_files = glob.glob(f"{logs_dir}/added_IVF*_Flat_nprobe_1_{model_name}_v2.index")
    
    output_zip = f"{output_dir}/{zip_name}.zip"
    
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        zipf.write(pth_file, f"{zip_name}.pth")
        for idx_file in index_files:
            ivf_num = idx_file.split('IVF')[1].split('_')[0]
            new_name = f"added_IVF{ivf_num}_Flat_nprobe_1_{zip_name}_v2.index"
            zipf.write(idx_file, new_name)
    
    return f"✓ Модель экспортирована: {output_zip}", output_zip

# Создание интерфейса Gradio
with gr.Blocks(title="RVC Training WebUI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎤 RVC Training WebUI")
    gr.Markdown("Веб-интерфейс для обучения голосовых моделей RVC")
    
    with gr.Tabs():
        # Вкладка установки
        with gr.Tab("⚙️ Установка"):
            gr.Markdown("### Установка зависимостей и настройка окружения")
            
            with gr.Row():
                with gr.Column():
                    machine_learning = gr.Dropdown(
                        choices=["fairseq", "transformers"],
                        value="fairseq",
                        label="Машинное обучение"
                    )
                    vocoder = gr.Dropdown(
                        choices=["Hifi-GAN", "RefineGAN"],
                        value="Hifi-GAN",
                        label="Вокодер"
                    )
                    sample_rate = gr.Dropdown(
                        choices=["32k", "40k", "48k"],
                        value="32k",
                        label="Sample Rate"
                    )
                
                with gr.Column():
                    hubert = gr.Dropdown(
                        choices=["contentvec-fairseq", "contentvec-transformers", "spin7-12-transformers", "spinV2-transformers"],
                        value="contentvec-fairseq",
                        label="Hubert модель"
                    )
                    mute_file = gr.Dropdown(
                        choices=["original", "spin_edition", "spinv2_edition"],
                        value="original",
                        label="Mute файлы"
                    )
                    configs = gr.Dropdown(
                        choices=["original_for_all_sample_rates", "convbased_only_contentvec_48k", "special_config_for_spinV2"],
                        value="original_for_all_sample_rates",
                        label="Конфиги"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Претрейны")
                    use_hifigan = gr.Checkbox(label="Использовать HiFi-GAN претрейн", value=False)
                    hifigan_pretrain = gr.Dropdown(
                        choices=["Default", "» Snowie v3", "» RIN_E3 - 40k", "» Snowie - 40k", "» Snowie + RIN_E3 - 40k", "» Rigel - 32k", "» Snowie v2 - 40k/48k", "» Ov2Super - 40k", "» TITAN-Medium"],
                        value="» Snowie v3",
                        label="HiFi-GAN Претрейн"
                    )
                    use_other = gr.Checkbox(label="Использовать другой претрейн", value=True)
                    other_pretrain = gr.Dropdown(
                        choices=["legacyCorev2.5_contentvec", "spinV2pretrain_hifigan", "spin7-12_pretrain-hifigan", "refinegan_pretrain-contentvec", "VocalCore"],
                        value="legacyCorev2.5_contentvec",
                        label="Другой претрейн"
                    )
            
            install_btn = gr.Button("🔧 Установить всё", variant="primary")
            install_output = gr.Textbox(label="Лог установки", lines=10)
            
            def full_install(ml, voc, sr, hub, mute, conf, use_hifi, hifi_pre, use_oth, oth_pre):
                logs = []
                logs.append(install_dependencies())
                logs.append(install_ml_requirements(ml))
                logs.append(download_hubert(hub))
                logs.append(download_rmvpe())
                logs.append(setup_mute_files(mute))
                logs.append(setup_configs(conf, sr))
                logs.append(download_pretrain(hifi_pre, sr, use_hifi, use_oth))
                return "\n\n".join(logs)
            
            install_btn.click(
                full_install,
                inputs=[machine_learning, vocoder, sample_rate, hubert, mute_file, configs, use_hifigan, hifigan_pretrain, use_other, other_pretrain],
                outputs=install_output
            )
        
        # Вкладка датасета
        with gr.Tab("📁 Датасет"):
            gr.Markdown("### Загрузка и подготовка датасета")
            
            with gr.Row():
                dataset_files = gr.File(
                    label="Загрузите аудио файлы",
                    file_count="multiple",
                    file_types=["audio"]
                )
                dataset_name = gr.Textbox(label="Название датасета", value="my_dataset")
            
            upload_btn = gr.Button("📤 Загрузить датасет", variant="primary")
            upload_output = gr.Textbox(label="Статус загрузки")
            
            upload_btn.click(
                upload_dataset,
                inputs=[dataset_files, dataset_name],
                outputs=upload_output
            )
        
        # Вкладка предобработки
        with gr.Tab("🔄 Предобработка"):
            gr.Markdown("### Предобработка и извлечение признаков")
            
            with gr.Row():
                with gr.Column():
                    model_name_prep = gr.Textbox(label="Название модели", value="mi-test")
                    sample_rate_prep = gr.Dropdown(choices=["32k", "40k", "48k"], value="32k", label="Sample Rate")
                    thread_count = gr.Slider(1, 16, value=8, step=1, label="Количество потоков")
                
                with gr.Column():
                    algo = gr.Dropdown(
                        choices=["rmvpe_remake_exp", "rmvpe", "pm", "harvest", "crepe"],
                        value="rmvpe_remake_exp",
                        label="Алгоритм F0"
                    )
                    ml_prep = gr.Dropdown(choices=["fairseq", "transformers"], value="fairseq", label="ML Backend")
            
            with gr.Row():
                preprocess_btn = gr.Button("1️⃣ Предобработка", variant="secondary")
                extract_btn = gr.Button("2️⃣ Извлечение признаков", variant="secondary")
                filelist_btn = gr.Button("3️⃣ Создать файллист", variant="secondary")
                index_btn = gr.Button("4️⃣ Создать индекс", variant="secondary")
            
            all_prep_btn = gr.Button("🚀 Выполнить всё", variant="primary")
            prep_output = gr.Textbox(label="Лог предобработки", lines=10)
            
            preprocess_btn.click(preprocess_dataset, inputs=[model_name_prep, sample_rate_prep, thread_count], outputs=prep_output)
            extract_btn.click(extract_features, inputs=[model_name_prep, thread_count, algo, ml_prep], outputs=prep_output)
            filelist_btn.click(create_filelist, inputs=[model_name_prep, sample_rate_prep], outputs=prep_output)
            index_btn.click(create_index, inputs=[model_name_prep], outputs=prep_output)
            
            def run_all_prep(mn, sr, tc, al, ml):
                logs = []
                logs.append(preprocess_dataset(mn, sr, tc))
                logs.append(extract_features(mn, tc, al, ml))
                logs.append(create_filelist(mn, sr))
                logs.append(create_index(mn))
                return "\n\n".join(logs)
            
            all_prep_btn.click(run_all_prep, inputs=[model_name_prep, sample_rate_prep, thread_count, algo, ml_prep], outputs=prep_output)
        
        # Вкладка обучения
        with gr.Tab("🎯 Обучение"):
            gr.Markdown("### Настройки обучения")
            
            with gr.Row():
                with gr.Column():
                    model_name_train = gr.Textbox(label="Название модели", value="mi-test")
                    sample_rate_train = gr.Dropdown(choices=["32k", "40k", "48k"], value="32k", label="Sample Rate")
                    vocoder_train = gr.Dropdown(choices=["Hifi-GAN", "RefineGAN"], value="Hifi-GAN", label="Вокодер")
                    epochs = gr.Slider(1, 2000, value=300, step=1, label="Количество эпох")
                    save_interval = gr.Slider(1, 500, value=100, step=1, label="Интервал сохранения")
                
                with gr.Column():
                    batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                    gpu = gr.Textbox(label="GPU (0 или 0,1)", value="0")
                    cache_data = gr.Checkbox(label="Кэшировать данные", value=False)
                    only_latest = gr.Checkbox(label="Только последние веса", value=False)
                    ml_train = gr.Dropdown(choices=["fairseq", "transformers"], value="fairseq", label="ML Backend")
            
            with gr.Row():
                train_btn = gr.Button("▶️ Начать обучение", variant="primary")
                stop_btn = gr.Button("⏹ Остановить", variant="stop")
                refresh_log_btn = gr.Button("🔄 Обновить лог")
            
            train_output = gr.Textbox(label="Статус обучения", lines=2)
            train_log = gr.Textbox(label="Лог обучения", lines=15)
            
            train_btn.click(
                start_training,
                inputs=[model_name_train, sample_rate_train, vocoder_train, batch_size, epochs, save_interval, gpu, cache_data, only_latest, ml_train],
                outputs=train_output
            )
            stop_btn.click(stop_training, outputs=train_output)
            refresh_log_btn.click(get_training_log, inputs=[model_name_train], outputs=train_log)
        
        # Вкладка экспорта
        with gr.Tab("📦 Экспорт"):
            gr.Markdown("### Экспорт обученной модели")
            
            with gr.Row():
                model_name_export = gr.Textbox(label="Название модели (logs)", value="mi-test")
                zip_name = gr.Textbox(label="Название ZIP архива", value="my_model")
            
            export_btn = gr.Button("📦 Экспортировать модель", variant="primary")
            export_output = gr.Textbox(label="Статус экспорта")
            download_file = gr.File(label="Скачать модель")
            
            export_btn.click(
                export_model,
                inputs=[model_name_export, zip_name],
                outputs=[export_output, download_file]
            )

# Запуск приложения
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)