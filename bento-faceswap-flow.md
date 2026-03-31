## Frontend Flow

┌─────────────────────────────────────────────────────────────┐
│  Bento Dashboard                                            │
│                                                             │
│  Content Studio → "Create Content" → Image Editor           │
│                                          │                  │
│                                    [Face Swap Tool] (later can add more tools)        │
│                                          │                  │
│  ┌───────────────────────────────────────────────────┐      │
│  │  Step 1: Upload / Select Images                   │      │
│  │  ┌───────────── ┐     ┌───────────── ┐            │      │
│  │  │ Source Face  │     │ Target Image │            │      │
│  │  │ (avatar)     │     │ (scene)      │            │      │
│  │  │ Upload / Pick│     │ Upload / Pick│            │      │
│  │  │ from library │     │ from library │            │      │
│  │  └───────────── ┘     └───────────── ┘            │      │
│  │                                                   │      │
│  │  Step 2: Preview & Confirm                        │      │
│  │  ┌──────────────────────────────────┐             │      │
│  │  │  [Detected faces highlighted]    │             │      │
│  │  │  Select which face to replace    │             │      │
│  │  │  [ Swap Now ]                    │             │      │
│  │  └──────────────────────────────────┘             │      │
│  │                                                   │      │
│  │  Step 3: Result                                   │      │
│  │  ┌──────────────────────────────────┐             │      │
│  │  │  Before / After slider           │             │      │
│  │  │  [Download] [Use in Post] [Redo] │             │      │
│  │  └──────────────────────────────────┘             │      │
│  └───────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘

## User Flow Summary
1. User opens Image Editor → clicks "Face Swap"
2. Picks source face (from Avatar Library or uploads new)
3. Uploads/selects target image
4. Faces auto-detected, highlighted with bounding boxes
5. User selects which faces to swap (if multiple)
6. Clicks "Swap" → job queued → real-time progress bar
7. Result shown with before/after slider
8. User clicks "Use in Post" → image attached to draft
9. User writes caption → schedules → publishes

## Also, we can keep a new tab called "Content Suggestion" beside "Advance AI Content Plan". We are already training over user's content. We need to identify page owner's face and page category, using them we can suggest some posts.


## Backend Architecture

                        ┌──────────────────┐
                        │   API Gateway    │
                        │   (FastAPI)      │
                        └────────┬─────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
         ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
         │ Media Service │ │ Swap Service│ │ Post Service│
         │              │ │            │ │             │
         │ - upload     │ │ - generate │ │ - draft     │
         │ - detect     │ │ - status   │ │ - schedule  │
         │   faces      │ │ - retry    │ │ - publish   │
         │ - store      │ │            │ │             │
         └──────┬───────┘ └─────┬──────┘ └─────────────┘
                │               │
         ┌──────▼───────┐ ┌─────▼──────────────────┐
         │ Cloudinary / │ │  Swap Worker (Queue)    │
         │ S3 Storage   │ │                         │
         └──────────────┘ │  Phase 1: Face detect   │
                          │  Phase 2: Face swap     │
                          │  Phase 3: Upscale       │
                          │  Phase 4: Enhance       │
                          │  Phase 5: Upload result │
                          └────────┬────────────────┘
                                   │
                          ┌────────▼────────────────┐
                          │  Model Inference         │
                          │                          │
                          │  Currently CPU (ONNX)    │
                          |  If volume increase, need GPU
                          └──────────────────────────┘




## Database Schema

media (This table needs if we are planning to keep/show/save media)
├── id              UUID PK
├── user_id         FK → users
├── url             TEXT              -- Cloudinary/S3 URL
├── public_id       TEXT              -- Cloudinary public_id (for deletion)
├── type            ENUM (image, video)
├── source          ENUM (upload, faceswap_result, ai_generated)
├── faces_detected  INT               -- cached count from detection
├── face_data       JSONB              -- [{bbox, landmarks, embedding_ref}, ...]
├── created_at      TIMESTAMP
└── deleted_at      TIMESTAMP          -- soft delete


avatars (reusable source faces)
├── id              UUID PK
├── user_id         FK → users
├── media_id        FK → media       
├── embedding       BYTEA
├── label           TEXT
└── created_at      TIMESTAMP

swap_jobs
├── id              UUID PK
├── user_id         FK → users
├── source_media_id FK → media         -- FK, not a string
├── target_media_id FK → media         -- FK
├── result_media_id FK → media         -- nullable, set on completion
├── source_face_idx INT DEFAULT 0
├── target_face_idx INT DEFAULT 0
├── status          ENUM (pending, processing, completed, failed)
├── current_phase   TEXT
├── error_message   TEXT
├── created_at      TIMESTAMP
├── started_at      TIMESTAMP
└── completed_at    TIMESTAMP

post_media (We may already have this table)
├── post_id         FK → posts
├── media_id        FK → media         -- FK
└── sort_order      INT


---

## API Detailed Breakdown (11 REST + 1 WebSocket)

---

### Media Service (3 endpoints)

#### `POST /api/media/upload`
Upload image + auto face detection in one call.

    Request:
      Content-Type: multipart/form-data
      Authorization: Bearer <token>
      Body:
        file: <binary>                    -- JPEG/PNG/WebP, max 10MB
        context: "faceswap_source"        -- or "faceswap_target", "post_attachment"

    Response: 201 Created
    {
      "media_id": "m_8f3a2b...",
      "url": "https://res.cloudinary.com/.../image.jpg",
      "type": "image",
      "width": 1024,
      "height": 768,
      "faces": [
        {
          "index": 0,
          "bbox": [120, 80, 340, 350],
          "confidence": 0.97,
          "thumbnail_url": "https://..."    -- cropped face for UI picker
        },
        {
          "index": 1,
          "bbox": [500, 100, 680, 320],
          "confidence": 0.91,
          "thumbnail_url": "https://..."
        }
      ]
    }

    Errors:
      400 -- unsupported format, corrupt image
      401 -- unauthorized
      413 -- file too large

    Server-side steps:
      1. Validate file type + size
      2. Upload to Cloudinary → get url + public_id
      3. If context is faceswap_* → run SCRFD face detection
      4. Cache face data (bbox, landmarks, embedding) in media.face_data JSONB
      5. Generate face thumbnail crops for UI face picker
      6. Insert row into media table
      7. Return response


#### `GET /api/media/{media_id}`
Fetch media metadata (used when selecting from library).

    Response: 200
    {
      "media_id": "m_8f3a2b...",
      "url": "https://...",
      "type": "image",
      "width": 1024,
      "height": 768,
      "source": "upload",
      "faces": [...],
      "created_at": "2026-03-25T10:30:00Z"
    }

    Errors:
      404 -- not found or not owned by user


#### `DELETE /api/media/{media_id}`
Soft-deletes media. Cloudinary cleanup runs via background job.

    Response: 204 No Content

    Errors:
      404 -- not found or not owned by user


---

### Avatar Library Service (3 endpoints)

#### `POST /api/avatars`
Save a detected face as a reusable avatar (caches embedding, skips re-detection on future swaps).

    Request:
    {
      "media_id": "m_8f3a2b...",
      "face_idx": 0,
      "label": "My headshot"
    }

    Response: 201
    {
      "avatar_id": "av_4c1d...",
      "label": "My headshot",
      "thumbnail_url": "https://...",
      "created_at": "2026-03-25T10:30:00Z"
    }

    Server-side:
      Extracts 512-dim ArcFace embedding from cached face_data
      and stores in avatars.embedding. Future swaps skip
      detection + embedding step entirely.


#### `GET /api/avatars`
List user's saved avatars for the UI picker.

    Query params: ?page=1&limit=20

    Response: 200
    {
      "avatars": [
        {
          "avatar_id": "av_4c1d...",
          "label": "My headshot",
          "thumbnail_url": "https://...",
          "created_at": "2026-03-25T10:30:00Z"
        }
      ],
      "total": 5
    }


#### `DELETE /api/avatars/{avatar_id}`

    Response: 204 No Content

    Errors:
      404 -- not found or not owned by user


---

### Face Swap Service (4 REST + 1 WebSocket)

#### `POST /api/faceswap/generate`
Kick off a swap job. Returns immediately with job ID.

    Request:
    {
      "source_media_id": "m_8f3a2b...",     -- use this OR avatar_id
      "source_avatar_id": "av_4c1d...",     -- optional, takes priority if set
      "target_media_id": "m_9d2c4e...",
      "source_face_idx": 0,
      "target_face_idx": 0
    }

    Response: 202 Accepted
    {
      "job_id": "j_7b3e1f...",
      "status": "pending",
      "created_at": "2026-03-25T10:31:00Z"
    }

    Errors:
      400 -- missing fields, no face at given index
      404 -- media/avatar not found
      429 -- daily swap quota exceeded

    Server-side:
      1. Validate media exists + face indices are valid
      2. Create swap_jobs row (status: pending)
      3. Push job to Redis/Celery queue
      4. Return job_id


#### `GET /api/faceswap/jobs/{job_id}`
Poll job status. Frontend calls every 2-3s (or use WebSocket instead).

    Response: 200

    While processing:
    {
      "job_id": "j_7b3e1f...",
      "status": "processing",
      "current_phase": "enhancing face",
      "phase_number": 4,
      "total_phases": 5,
      "started_at": "2026-03-25T10:31:01Z"
    }

    On completion:
    {
      "job_id": "j_7b3e1f...",
      "status": "completed",
      "result_media_id": "m_f1a2b3...",
      "result_url": "https://res.cloudinary.com/.../swapped.jpg",
      "original_url": "https://...target.jpg",
      "duration_seconds": 42.3,
      "completed_at": "2026-03-25T10:31:43Z"
    }

    On failure:
    {
      "job_id": "j_7b3e1f...",
      "status": "failed",
      "error_message": "No face detected in target image",
      "can_retry": true
    }


#### `WS /ws/faceswap/jobs/{job_id}`
WebSocket for real-time phase updates (alternative to polling).

    Server pushes:
    { "phase": "detecting faces",  "phase_number": 1, "total": 5 }
    { "phase": "swapping face",    "phase_number": 2, "total": 5 }
    { "phase": "upscaling image",  "phase_number": 3, "total": 5 }
    { "phase": "enhancing face",   "phase_number": 4, "total": 5 }
    { "phase": "uploading result", "phase_number": 5, "total": 5 }
    { "phase": "done", "status": "completed", "result_url": "https://..." }


#### `POST /api/faceswap/jobs/{job_id}/retry`
Retry a failed job. Reuses cached phase results if any phases completed.

    Response: 202 Accepted
    {
      "job_id": "j_7b3e1f...",
      "status": "pending"
    }

    Errors:
      400 -- job is not in failed state
      404 -- job not found


#### `GET /api/faceswap/history`
User's swap history (for "Recent Swaps" in UI).

    Query params: ?page=1&limit=20&status=completed

    Response: 200
    {
      "jobs": [
        {
          "job_id": "j_7b3e1f...",
          "status": "completed",
          "result_url": "https://...",
          "source_thumbnail": "https://...",
          "target_thumbnail": "https://...",
          "created_at": "2026-03-25T10:31:00Z"
        }
      ],
      "total": 47,
      "page": 1,
      "limit": 20
    }


---

### Content Suggestion Service (1 endpoint)

#### `GET /api/suggestions/faceswap`
Uses page owner's face + page category to suggest swap templates.

    Query params: ?page_id=pg_123&limit=5

    Response: 200
    {
      "suggestions": [
        {
          "template_id": "tmpl_seasonal_01",
          "preview_url": "https://...",
          "category": "seasonal_greeting",
          "caption_suggestion": "Happy Holi from...",
          "target_image_url": "https://..."
        }
      ]
    }

    Server-side:
      Matches page category (food, fashion, fitness, etc.)
      + current trends/season → picks template images
      → user one-clicks to swap their face into the template


---

### API Summary

    Service              Endpoints    Methods
    ──────────────────────────────────────────
    Media                3            POST, GET, DELETE
    Avatars              3            POST, GET, DELETE
    Face Swap            4 + 1 WS     POST, GET, POST, GET + WS
    Suggestions          1            GET
    ──────────────────────────────────────────
    Total                11 + 1 WS
