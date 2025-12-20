import React, { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import { GoldenImage } from '../types'
import './AdminPanel.css'

function AdminPanel() {
  const [templates, setTemplates] = useState<GoldenImage[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [templateName, setTemplateName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [feedback, setFeedback] = useState('')

  useEffect(() => {
    loadTemplates()
  }, [])

  const loadTemplates = async () => {
    try {
      setLoading(true)
      const { data, error } = await supabase
        .from('golden_images')
        .select('*')
        .order('created_at', { ascending: false })

      if (error) throw error
      setTemplates(data || [])
    } catch (error) {
      console.error('Failed to load templates:', error)
      setFeedback('Failed to load templates')
    } finally {
      setLoading(false)
    }
  }

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!file || !templateName.trim()) {
      setFeedback('Please fill in all fields')
      return
    }

    try {
      setUploading(true)
      setFeedback('')

      const fileName = `golden/${Date.now()}_${file.name}`
      const { error: uploadError } = await supabase.storage
        .from('images')
        .upload(fileName, file)

      if (uploadError) throw uploadError

      const { data: publicUrl } = supabase.storage
        .from('images')
        .getPublicUrl(fileName)

      const { error: dbError } = await supabase.from('golden_images').insert({
        name: templateName.trim(),
        image_path: publicUrl.publicUrl,
        metadata: {
          uploaded_at: new Date().toISOString(),
          file_size: file.size,
          file_type: file.type,
        },
      })

      if (dbError) throw dbError

      setFeedback('Template uploaded successfully!')
      setTemplateName('')
      setFile(null)
      loadTemplates()
    } catch (error) {
      console.error('Upload failed:', error)
      setFeedback('Upload failed. Please try again.')
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this template?')) return

    try {
      const { error } = await supabase.from('golden_images').delete().eq('id', id)

      if (error) throw error

      setFeedback('Template deleted successfully')
      loadTemplates()
    } catch (error) {
      console.error('Delete failed:', error)
      setFeedback('Failed to delete template')
    }
  }

  if (loading) {
    return (
      <div className="admin-panel">
        <div className="loading">Loading...</div>
      </div>
    )
  }

  return (
    <div className="admin-panel">
      <h1>Golden Templates Manager</h1>

      <div className="admin-content">
        <div className="upload-section">
          <h2>Upload Master Image</h2>
          <form onSubmit={handleUpload} className="upload-form">
            <div className="form-group">
              <label>Template Name</label>
              <input
                type="text"
                value={templateName}
                onChange={(e) => setTemplateName(e.target.value)}
                placeholder="e.g., Product Model A v1"
                disabled={uploading}
              />
            </div>

            <div className="form-group">
              <label>Image File</label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                disabled={uploading}
              />
            </div>

            {feedback && <div className="feedback">{feedback}</div>}

            <button type="submit" disabled={uploading || !file || !templateName} className="upload-button">
              {uploading ? 'Uploading...' : 'Upload Template'}
            </button>
          </form>
        </div>

        <div className="templates-section">
          <h2>Existing Templates ({templates.length})</h2>
          <div className="templates-grid">
            {templates.length === 0 ? (
              <p className="empty-state">No templates uploaded yet</p>
            ) : (
              templates.map((template) => (
                <div key={template.id} className="template-card">
                  <img src={template.image_path} alt={template.name} className="template-image" />
                  <div className="template-info">
                    <h3>{template.name}</h3>
                    <p className="template-date">
                      {new Date(template.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button
                    onClick={() => handleDelete(template.id)}
                    className="delete-button"
                  >
                    Delete
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default AdminPanel
