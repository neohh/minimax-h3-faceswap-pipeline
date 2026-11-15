using System;
using System.IO;
using System.Windows.Forms;

namespace SendToSammie
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            string filePath = null;
            if (args.Length > 0 && !string.IsNullOrEmpty(args[0]))
            {
                filePath = args[0];
            }
            else
            {
                using (OpenFileDialog ofd = new OpenFileDialog())
                {
                    ofd.Title = "Выберите видео для загрузки в Sammie-Roto";
                    ofd.Filter = "Видео файлы (*.mp4;*.mov;*.avi;*.mkv;*.png;*.jpg)|*.mp4;*.mov;*.avi;*.mkv;*.png;*.jpg|Все файлы (*.*)|*.*";
                    if (ofd.ShowDialog() == DialogResult.OK)
                    {
                        filePath = ofd.FileName;
                    }
                }
            }

            if (!string.IsNullOrEmpty(filePath) && File.Exists(filePath))
            {
                try
                {
                    string tempDir = @"Z:\Work\VFX_Program\Sammie-Roto-2-main\temp";
                    Directory.CreateDirectory(tempDir);
                    string ipcFile = Path.Combine(tempDir, "sammie_ipc_command.json");
                    string json = "{\"action\": \"load_file\", \"file\": \"" + filePath.Replace("\\", "\\\\") + "\"}";
                    File.WriteAllText(ipcFile, json, System.Text.Encoding.UTF8);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Ошибка отправки: " + ex.Message, "Sammie-Roto Loader", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }
    }
}
