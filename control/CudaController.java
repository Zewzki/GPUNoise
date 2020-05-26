package control;

import static jcuda.driver.JCudaDriver.*;

import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.util.Random;

import javax.swing.JFrame;

import jcuda.*;
import jcuda.driver.*;

public class CudaController extends JFrame {

	private CudaPanel cp;

	private boolean running;
	private Thread thread;

	private static final int WIDTH = 1024;
	private static final int HEIGHT = 1024;

	private int[] pixelVals;

	private double xPos;
	private double yPos;
	private double zPos;

	private Random rand;

	private String ptxFileName;

	private static CUdevice device;
	private static CUcontext context;
	private static CUmodule module;
	private static CUfunction function;
	private static CUdeviceptr output;

	private int numOfElements;

	private Simplex simplex;

	public CudaController() {

		setResizable(false);
		setSize(WIDTH, HEIGHT);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		setLocationRelativeTo(null);
//		addWindowListener(new WindowListener() {
//
//			@Override
//			public void windowActivated(WindowEvent arg0) {}
//
//			@Override
//			public void windowClosed(WindowEvent arg0)  {}
//
//			@Override
//			public void windowClosing(WindowEvent arg0) {
//				
//				dispose();
//				shutdown();
//				System.exit(0);
//				
//			}
//
//			@Override
//			public void windowDeactivated(WindowEvent arg0) {}
//
//			@Override
//			public void windowDeiconified(WindowEvent arg0) {}
//
//			@Override
//			public void windowIconified(WindowEvent e) {}
//
//			@Override
//			public void windowOpened(WindowEvent e) {}
//			
//		});

		cp = new CudaPanel(WIDTH, HEIGHT);

		add(cp);

		setVisible(true);
		
		xPos = 0;
		yPos = 0;

		simplex = new Simplex();
		pixelVals = new int[WIDTH * HEIGHT];

		numOfElements = WIDTH * HEIGHT;
		
	}

	public synchronized void start() {
		if (running) {
			return;
		}

		running = true;
		
		initCuda(false);
		
		xPos = 100;
		yPos = 100;
		
		run();

	}

	public void run() {
		
		cuMemcpyHtoD(output, Pointer.to(pixelVals), numOfElements * Sizeof.INT);
		
		long initialTime = System.nanoTime();
        double timeF = 1000000000 / 60;
        double deltaF = 0;
        int frames = 0;
        long timer = System.currentTimeMillis();
		
		while (running) {
			
			long currentTime = System.nanoTime();
            deltaF += (currentTime - initialTime) / timeF;
            initialTime = currentTime;

            if(deltaF >= 1) {
                
            	Pointer kernelParameters = Pointer.to(
    					Pointer.to(output),
    					Pointer.to(new double[] {xPos}),
    					Pointer.to(new double[] {yPos}),
    					Pointer.to(new double[] {zPos}),
    					Pointer.to(new int[] {WIDTH})
    			);

    			int blockSizeX = 16;
    			int blockSizeY = 16;
    			int blockSizeZ = 1;
    			int gridSizeX = 64;
    			int gridSizeY = 64;
    			int gridSizeZ = 1;
    			
    			//int gridSizeX = (int) Math.ceil((double) numOfElements / blockSizeX);
    			cuLaunchKernel(function, gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ, 0, null, kernelParameters, null);

    			cuCtxSynchronize();
    			
    			cuMemcpyDtoH(Pointer.to(pixelVals), output, numOfElements * Sizeof.INT);
    			
    			cp.update(pixelVals);
    			cp.repaint();
            	
    			xPos += 1;
    			yPos += 1;
    			//zPos += 2;
            	
                frames++;
                deltaF--;
            }

            if(System.currentTimeMillis() - timer > 1000) {
                //System.out.println("FPS: " + frames);
                frames = 0;
                timer += 1000;
            }

		}
		
		shutdown();

	}

	public void initCuda(boolean prepareKernel) {
		cuInit(0);
		device = new CUdevice();
		cuDeviceGet(device, 0);
		context = new CUcontext();
		cuCtxCreate(context, 0, device);
		
		setExceptionsEnabled(true);
		
		prepareKernel();
	}

	public void prepareKernel() {
		
		//ptxFileName = JCudaSamplesUtils.preparePtxFile("C:\\Users\\Michael\\eclipse\\java-neon\\eclipse\\Java\\workspace\\GPUNoise\\src\\kernels\\noise.cu");
		//ptxFileName = JCudaSamplesUtils.preparePtxFile("C:\\Users\\Michael\\eclipse\\java-neon\\eclipse\\Java\\workspace\\GPUNoise\\src\\kernels\\perlin.cu");
		//ptxFileName = JCudaSamplesUtils.preparePtxFile("C:\\Users\\Michael\\eclipse\\java-neon\\eclipse\\Java\\workspace\\GPUNoise\\src\\kernels\\sumOctave.cu");
		
		ptxFileName = "C:\\Users\\Michael\\eclipse\\java-neon\\eclipse\\Java\\workspace\\GPUNoise\\src\\kernels\\sumOctave.ptx";
		
		module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		function = new CUfunction();
		//cuModuleGetFunction(function, module, "noise");
		//cuModuleGetFunction(function, module, "random");
		cuModuleGetFunction(function, module, "sumOctave");

		output = new CUdeviceptr();
		cuMemAlloc(output, numOfElements * Sizeof.INT);

	}

	public void shutdown() {
		cuModuleUnload(module);
		cuMemFree(output);
		if (context != null) {
			cuCtxDestroy(context);
		}
	}

	public static void main(String[] args) {

		JCudaDriver.setExceptionsEnabled(true);

		CudaController cc = new CudaController();

		cc.start();
		
	}

}
