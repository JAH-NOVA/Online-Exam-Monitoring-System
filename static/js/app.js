// JavaScript for upload functionality and progress tracking

// Drag and drop functionality
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('videoFile');
const uploadForm = document.getElementById('uploadForm');
const fileInfo = document.getElementById('fileInfo');
const progressContainer = document.getElementById('progressContainer');

if (uploadZone && fileInput) {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadZone.classList.add('drag-over');
    }

    function unhighlight() {
        uploadZone.classList.remove('drag-over');
    }

    // Handle dropped files
    uploadZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    }

    // Handle file selection from input
    fileInput.addEventListener('change', handleFileSelect);

    function handleFileSelect() {
        const file = fileInput.files[0];
        
        if (!file) return;
        
        // Validate file type
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-flv', 'video/x-ms-wmv'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|avi|mov|mkv|flv|wmv)$/i)) {
            alert('Invalid file type. Please upload a video file (MP4, AVI, MOV, MKV, FLV, WMV)');
            return;
        }
        
        // Validate file size (500MB = 524288000 bytes)
        const maxSize = 524288000;
        if (file.size > maxSize) {
            alert('File is too large. Maximum size is 500MB');
            return;
        }
        
        // Display file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = formatFileSize(file.size);
        fileInfo.style.display = 'block';
        uploadZone.style.display = 'none';
    }

    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            // Show progress container
            fileInfo.style.display = 'none';
            progressContainer.classList.add('active');
            
            // Upload file
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                document.getElementById('statusMessage').textContent = 'Uploading video...';
                
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Upload failed');
                }
                
                const result = await response.json();
                const jobId = result.job_id;
                
                // Start polling for progress
                document.getElementById('statusMessage').textContent = 'Processing video...';
                pollProgress(jobId);
                
            } catch (error) {
                console.error('Upload error:', error);
                alert('Upload failed. Please try again.');
                progressContainer.classList.remove('active');
                uploadZone.style.display = 'block';
            }
        });
    }
}

// Poll for processing progress
function pollProgress(jobId) {
    const pollInterval = setInterval(async () => {
        try {
            const response = await fetch(`/status/${jobId}`);
            
            if (!response.ok) {
                throw new Error('Failed to get status');
            }
            
            const status = await response.json();
            
            // Update progress bar
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            if (progressFill && progressText) {
                progressFill.style.width = `${status.progress}%`;
                progressText.textContent = `${status.progress}% Complete`;
            }
            
            // Update status message
            const statusMessage = document.getElementById('statusMessage');
            if (statusMessage) {
                if (status.status === 'processing') {
                    statusMessage.textContent = `Processing frame ${status.current_frame} of ${status.total_frames}...`;
                } else if (status.status === 'completed') {
                    statusMessage.textContent = 'Processing complete! Redirecting...';
                    clearInterval(pollInterval);
                    
                    // Redirect to results page
                    setTimeout(() => {
                        window.location.href = `/results/${jobId}`;
                    }, 1000);
                } else if (status.status === 'error') {
                    statusMessage.textContent = `Error: ${status.error}`;
                    clearInterval(pollInterval);
                    alert(`Processing failed: ${status.error}`);
                }
            }
            
        } catch (error) {
            console.error('Status check error:', error);
            clearInterval(pollInterval);
            alert('Failed to check processing status');
        }
    }, 2000); // Check every 2 seconds
}

// Helper function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}
