package org.gpucache.cache;

/**
 * Column interface
 * @author Fabrice
 *
 * @param <T>
 */
public interface ColumnI {
	
	public ColumnI init(String name, int initialcapacity);
	
	/**
	 * Gives the column name
	 * @return
	 */
	public String getName();
	
	/**
	 * Gives the number of elements
	 * @return
	 */
	public int count();
	
	/**
	 * Copy the content into GPU memory
	 * @return
	 */
	public boolean copyToGPU();
	
	/**
	 * Return an integer 4 bytes
	 * @param index
	 * @return
	 */
	public int getInt(int  index);

	
	public float getFloat(int index);
	
	public void addInt(int value);
	
	public void addFloat(float value);
	
	public boolean remove(int index);
	
	
}
