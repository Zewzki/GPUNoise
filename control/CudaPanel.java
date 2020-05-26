package control;

import java.awt.Color;
import java.awt.Graphics;
import java.util.Random;

import javax.swing.JPanel;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.IOException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;


public class CudaPanel extends JPanel {
	
	private int[][] pixelVals;
	
	private int width;
	private int height;
	
	public CudaPanel(int w, int h) {
		
		pixelVals = new int[w][h];
		width = w;
		height = h;
		
	}
	
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		
//		for(int i = 0; i < width; i++) {
//			for(int j = 0; j < height; j++) {
//				System.out.print(pixelVals[(i * width) + j] + ", ");
//			}
//			System.out.println();
//		}
		
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				
				try {
					g.setColor(new Color(pixelVals[i][j], pixelVals[i][j], pixelVals[i][j]));
					//System.out.println(pixelVals[i][j]);
				} catch (IllegalArgumentException e) {
					//System.err.println("Out of Bounds: " + pixelVals[i][j]);
					g.setColor(Color.GREEN);
					//System.exit(1);
				}
				g.fillRect(i, j, 1, 1);
			}
		}
		
	}
	
	public void update(int[] newPixels) {
		
		for(int i = 0; i < width; i++) {
			for(int j = 0; j < height; j++) {
				pixelVals[i][j] = newPixels[(i * width) + j];
			}
		}
		
		
	}
	
}
