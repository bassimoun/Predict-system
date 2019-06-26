using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;

namespace Ultrasound_reg
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }
        private void background_python(string code)
        {

            System.IO.StreamReader file = new System.IO.StreamReader(@"Python.txt");
            string line = file.ReadLine();
            line.Replace("\\", "\\\\");
            Process p = new Process();

            p.StartInfo = new ProcessStartInfo(@line, code)
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                RedirectStandardError = true,
                CreateNoWindow = true
            };
            p.Start();
            p.WaitForExit();
            p.Close();
        }
        
        private void Upload_Img_Click(object sender, EventArgs e)
        {
            textBox1.Text = "";
            textBox2.Text = "";
            OpenFileDialog opnfd = new OpenFileDialog();
            opnfd.Filter = "Image Files (*.png;*.jpg;*.jpeg;.*.gif;)|*.png;*.jpg;*.jpeg;.*.gif";
            if (opnfd.ShowDialog() == DialogResult.OK)
            {
                string Dir = opnfd.FileName;
                Bitmap img = (Bitmap)Image.FromFile(Dir, true);
                img.Save(@"non_space_name_img.jpg");
                pictureBox1.Image = new Bitmap(opnfd.FileName);
                pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;
            }
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            string code_arguments = "cnn.py non_space_name_img.jpg";
            background_python(code_arguments);
            int counter = 0;
            string line;
            System.IO.StreamReader file = new System.IO.StreamReader(@"ces_normal_prob.txt");
            while ((line = file.ReadLine()) != null)
            {
                if (counter == 0)
                {
                    textBox1.Text = line;
                }
                if (counter == 1)
                {
                    textBox2.Text = line;
                }
                counter += 1;
            }
            file.Close();
        }
    }
}
