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

namespace Ces_Norm_Classification
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        private string background_python(string code)
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

            string output = p.StandardOutput.ReadToEnd();
            string stderr = p.StandardError.ReadToEnd();
            
            p.Close();
            return output;
        }
        
        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void numericUpDown1_ValueChanged(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox4_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox3_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox6_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox5_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox7_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox8_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox9_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void comboBox10_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        private void Check_Case_Click(object sender, EventArgs e)
        {
            float f;
            if ((textBox1.Text == "") || (numericUpDown1.Value == 0) || (comboBox1.Text == "") || (comboBox2.Text == "") || (comboBox3.Text == "") || (comboBox4.Text == "") || (comboBox5.Text == "") || (comboBox6.Text == "") || (comboBox7.Text == "") || (comboBox8.Text == "") || (comboBox9.Text == "") || (comboBox10.Text == ""))
            {
                MessageBox.Show("Please fill all attributes...\n Thanks.");
            }
            else
            {
                if (float.TryParse(textBox1.Text, out f))
                {
                    if ((f >= 15.0) || (f <= 8.0))
                    {
                        MessageBox.Show("Please set HB GM/D as float number\n from 8 to 15 \n then try again");
                    }
                    else
                    {
                        if (comboBox11.Text == "SVM")
                        {
                            string code_arguments = "";
                            code_arguments = "eval_svm.py " + textBox1.Text + " " + Convert.ToString(numericUpDown1.Value) + " " + comboBox1.Text + " " + comboBox2.Text + " " + comboBox3.Text + " " + comboBox4.Text + " " + comboBox5.Text + " " + comboBox6.Text + " " + comboBox7.Text + " " + comboBox8.Text + " " + comboBox9.Text + " " + comboBox10.Text;
                            string class_birth = background_python(code_arguments);
                            if (class_birth.Substring(1, 1) == "0")
                            {
                                textBox2.Text = "NORMAL BIRTH";
                            }
                            else
                            {
                                textBox2.Text = "CESAREAN BIRTH";
                            }
                        }
                        else if (comboBox11.Text == "ANN")
                        {
                            string code_arguments = "";
                            code_arguments = "eval_ANN.py " + textBox1.Text + " " + Convert.ToString(numericUpDown1.Value) + " " + comboBox1.Text + " " + comboBox2.Text + " " + comboBox3.Text + " " + comboBox4.Text + " " + comboBox5.Text + " " + comboBox6.Text + " " + comboBox7.Text + " " + comboBox8.Text + " " + comboBox9.Text + " " + comboBox10.Text;
                            string class_birth = background_python(code_arguments);
                            if (class_birth.Substring(1, 1) == "0")
                            {
                                textBox2.Text = "NORMAL BIRTH";
                            }
                            else
                            {
                                textBox2.Text = "CESAREAN BIRTH";
                            }
                        }
                        else
                        {
                            MessageBox.Show("Please set Classifier and try again");

                        }

                    }
                }
                else
                {
                    MessageBox.Show("Please set HB GM/D as float number and try again");
                }
            }
            

            
        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
