package test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.UInt1Vector;
import org.apache.arrow.vector.UInt4Vector;
import org.junit.Before;
import org.junit.jupiter.api.Test;

import io.netty.buffer.ArrowBuf;
import io.netty.buffer.ByteBuf;

class ArrowUnitTest {
	private final static String EMPTY_SCHEMA_PATH = "";

	  private BufferAllocator allocator;

	  @Before
	  public void init() {
	    allocator = new RootAllocator(Long.MAX_VALUE);
	  }
	 
	  public void testFixedType1() {
		  allocator = new RootAllocator(Long.MAX_VALUE);
	    // Create a new value vector for 1024 integers.
	    try (final UInt1Vector vector = new UInt1Vector(EMPTY_SCHEMA_PATH, allocator)) {

	      boolean error = false;
	      int initialCapacity = 0;
	      final UInt1Vector.Mutator mutator = vector.getMutator();
	      final UInt1Vector.Accessor accessor = vector.getAccessor();

	      vector.allocateNew(1024);
	      initialCapacity = vector.getValueCapacity();
	      assertEquals(1024, initialCapacity);

	      // Put and set a few values
	      mutator.setSafe(0, 100);
	      mutator.setSafe(1, 101);
	      mutator.setSafe(100, 102);
	      mutator.setSafe(122, 103);
	      mutator.setSafe(123, 104);

	      assertEquals(100, accessor.get(0));
	      assertEquals(101, accessor.get(1));
	      assertEquals(102, accessor.get(100));
	      assertEquals(103, accessor.get(122));
	      assertEquals(104, accessor.get(123));

	      try {
	        mutator.set(1024, 100);
	      }
	      catch (IndexOutOfBoundsException ie) {
	        error = true;
	      }
	      finally {
	        assertTrue(error);
	        error = false;
	      }

	      try {
	        accessor.get(1024);
	      }
	      catch (IndexOutOfBoundsException ie) {
	        error = true;
	      }
	      finally {
	        assertTrue(error);
	        error = false;
	      }

	      /* this should trigger a realloc() */
	      mutator.setSafe(1024, 100);

	      /* underlying buffer should now be able to store double the number of values */
	      assertEquals(initialCapacity * 2, vector.getValueCapacity());

	      /* check vector data after realloc */
	      assertEquals(100, accessor.get(0));
	      assertEquals(101, accessor.get(1));
	      assertEquals(102, accessor.get(100));
	      assertEquals(103, accessor.get(122));
	      assertEquals(104, accessor.get(123));
	      assertEquals(100, accessor.get(1024));

	      /* reset the vector */
	      vector.reset();

	      /* capacity shouldn't change after reset */
	      assertEquals(initialCapacity * 2, vector.getValueCapacity());

	      /* vector data should have been zeroed out */
	     
	    }
	  }
	  @Test /* UInt4Vector */
	  public void testFixedType4() {
		  allocator = new RootAllocator(Long.MAX_VALUE);
	    // Create a new value vector for 1024 integers.
	    try (final UInt4Vector vector = new UInt4Vector(EMPTY_SCHEMA_PATH, allocator)) {

	      boolean error = false;
	      int initialCapacity = 0;
	      final UInt4Vector.Mutator mutator = vector.getMutator();
	      final UInt4Vector.Accessor accessor = vector.getAccessor();
	      int size = 1<<24;
	      System.out.println(1<<8);
	      System.out.println(1<<16);
	      System.out.println(1<<24);
	      System.out.println(1<<31);
	      vector.allocateNew(size);
	      initialCapacity = vector.getValueCapacity();
	     
	      // Put and set a few values
	      for (int i = 0; i < size; i++) {
	    	  mutator.setSafe(i, i);
	      }
	      /* vector data should have been zeroed out */
	      long sum = 0;
	      long start = System.currentTimeMillis();
	      for (int i = 0; i < size; i++) {
	    	  sum += accessor.get(i);
	      }
	      long stop = System.currentTimeMillis();
	      long valeur =  size/2;
	      valeur = valeur * (size-1);
	      System.out.println(sum +" ="+valeur);
	      System.out.println(stop-start);
	    }
	  }
	  
	  @Test /* UInt4Vector */
	  public void testFixedCuda() {
		  allocator = new RootAllocator(Long.MAX_VALUE);
	    // Create a new value vector for 1024 integers.
	    try (final UInt4Vector vector = new UInt4Vector(EMPTY_SCHEMA_PATH, allocator)) {

	      boolean error = false;
	      int initialCapacity = 0;
	      final UInt4Vector.Mutator mutator = vector.getMutator();
	      final UInt4Vector.Accessor accessor = vector.getAccessor();
	      int size = 1<<4;
	      
	      vector.allocateNew(size);
	      initialCapacity = vector.getValueCapacity();
	     
	      // Put and set a few values
	      for (int i = 0; i < size; i++) {
	    	  mutator.setSafe(i, i);
	      }
	      /* vector data should have been zeroed out */
	      long sum = 0;
	      long start = System.currentTimeMillis();
	      for (int i = 0; i < size; i++) {
	    	  sum += accessor.get(i);
	      }
	      long stop = System.currentTimeMillis();
	      long valeur =  size/2;
	      valeur = valeur * (size-1);
	      System.out.println(sum +" ="+valeur);
	      System.out.println(stop-start);
	      byte[] valeurs = new byte[size*4*16];
	     ArrowBuf buf =vector.getBuffer();
	     vector.getBuffer().memoryAddress();
	     byte[] bytes = new byte[buf.readableBytes()];
	     buf.readBytes(bytes);
	     for (int i = 0; i < size; i++) {
	    	System.out.println(valeurs[i]+"-" +   vector.getBuffer().memoryAddress());
	      }
	    }
	  }
}
