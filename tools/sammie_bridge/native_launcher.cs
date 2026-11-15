using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace SammieLauncher
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            try
            {
                string workDir = @"Z:\Work\VFX_Program\Sammie-Roto-2-main";
                ProcessStartInfo psi = new ProcessStartInfo();
                psi.FileName = "cmd.exe";
                string argStr = args.Length > 0 ? ("\"" + string.Join("\" \"", args) + "\"") : "";
                psi.Arguments = "/c run_sammie.bat " + argStr;
                psi.WorkingDirectory = workDir;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                Process.Start(psi);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error launching Sammie-Roto: " + ex.Message, "Sammie-Roto Launcher", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
    }
}
